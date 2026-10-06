import json
from pathlib import Path

from harness import db, grid


def test_nondominated_fronts_simple_case():
    # (x, y), both minimized. (1,1) dominates everything. (2,0.5) and (0.5,2)
    # are mutually non-dominated with each other and with (1,1)'s dominated
    # neighbors, but NOT with (1,1) itself.
    points = [(1, 1), (2, 2), (0.5, 2), (2, 0.5), (3, 3)]
    fronts = grid.nondominated_fronts(points)
    assert fronts[0] == 0  # (1,1) is non-dominated
    assert fronts[1] > 0  # (2,2) is dominated by (1,1)
    assert fronts[4] > fronts[1]  # (3,3) is dominated by (2,2) too -- a later front


def test_nondominated_fronts_front_zero_includes_tradeoff_extremes():
    # The whole point: the cheapest/worst-quality corner must NOT be
    # excluded just because something else has better quality, as long as
    # nothing beats it on BOTH axes. (kl_mean, predicted_mb) both minimized.
    points = [
        (0.01, 5000),   # best quality, most memory
        (0.50, 1500),   # worst quality, least memory -- still non-dominated
        (0.10, 3000),   # a middle point
        (0.60, 1600),   # dominated by (0.50, 1500): worse on both axes
    ]
    fronts = grid.nondominated_fronts(points)
    assert fronts[0] == 0
    assert fronts[1] == 0  # the cheap/low-quality extreme survives front 0
    assert fronts[3] > 0   # strictly dominated


def test_nondominated_fronts_empty():
    assert grid.nondominated_fronts([]) == []


def test_fa_all_quants_enabled_parses_on(tmp_path):
    cache = tmp_path / "CMakeCache.txt"
    cache.write_text("GGML_CUDA_FA_ALL_QUANTS:BOOL=ON\nSOME_OTHER:BOOL=OFF\n")
    bin_dir = tmp_path / "build" / "bin"
    bin_dir.mkdir(parents=True)
    assert grid.fa_all_quants_enabled(str(bin_dir)) is True


def test_fa_all_quants_enabled_parses_off(tmp_path):
    cache = tmp_path / "CMakeCache.txt"
    cache.write_text("GGML_CUDA_FA_ALL_QUANTS:BOOL=OFF\n")
    bin_dir = tmp_path / "build" / "bin"
    bin_dir.mkdir(parents=True)
    assert grid.fa_all_quants_enabled(str(bin_dir)) is False


def test_fa_all_quants_enabled_missing_file_is_conservative(tmp_path):
    bin_dir = tmp_path / "build" / "bin"
    bin_dir.mkdir(parents=True)
    assert grid.fa_all_quants_enabled(str(bin_dir)) is False


def _insert_gate_row(con, spec_hash, artifact_hash, kl_mean, ctx, kv_k, kv_v, size_bytes, path,
                      run_id, eval_name="kld_c512_n20", split="search"):
    con.execute("INSERT OR IGNORE INTO specs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?, current_timestamp)",
                [spec_hash, artifact_hash, "m.gguf", "Q4_K_M", json.dumps({}), None, kv_k, kv_v,
                 True, ctx, None, None, json.dumps([]), json.dumps({})])
    con.execute("INSERT OR IGNORE INTO artifacts VALUES (?,?,?,?,?,?,?,?)",
                [artifact_hash, path, size_bytes, 3_000_000_000, 4.9, 10.0, "cmd", "2026-10-01T00:00:00"])
    con.execute("INSERT OR IGNORE INTO environments VALUES (?,?,?,?,?,?,?,?)",
                ["env1", "RTX 3080 Ti", 12288, "596.36", "13.4", "abc123", 400.0, None])
    wl = f"wl_{run_id}"
    con.execute("INSERT OR IGNORE INTO workloads VALUES (?,?,?,?,?,?,?)",
                [wl, "kld", None, ctx, 0, 1, json.dumps({})])
    con.execute("INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [run_id, spec_hash, "env1", wl, 0, None, 60, 1800, "2026-10-01T00:00:00", 1.0,
                 "ok", None, None])
    con.execute("INSERT OR REPLACE INTO quality VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                [run_id, eval_name, split, None, kl_mean, None, None, None, None, None, None])


def test_promote_includes_baseline_and_writes_survivor_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    con = db.connect(str(tmp_path / "t.duckdb"))
    # A real GGUF is needed for memory.shape_from_gguf -- use the project's
    # own Q4_K_M file if present, else skip this part via a stub path. Here
    # we fake it by monkeypatching shape_from_gguf's dependency chain is
    # overkill for a unit test; instead exercise promote() against a tiny
    # fake "artifact" whose path doesn't need to resolve (shape_from_gguf is
    # the one real-file dependency) by monkeypatching harness.memory calls.
    import harness.grid as grid_mod

    def fake_shape_from_gguf(path):
        from harness.memory import ModelShape
        return ModelShape(kv_layers=32, kv_heads=2, key_dim=128, value_dim=128)

    monkeypatch.setattr(grid_mod.memory, "shape_from_gguf", fake_shape_from_gguf)

    _insert_gate_row(con, "spec_good", "art_good", 0.01, 32768, "f16", "f16", 3_000_000_000,
                      "x.gguf", "run_good")
    _insert_gate_row(con, "spec_cheap", "art_cheap", 0.50, 32768, "q4_0", "q4_0", 1_500_000_000,
                      "y.gguf", "run_cheap")
    _insert_gate_row(con, "spec_baseline", "art_base", 0.10, 32768, "q8_0", "q8_0", 2_000_000_000,
                      "z.gguf", "run_base")

    result = grid.promote(con, "w5test", "kld_c512_n20", "search",
                           baseline_spec_hash="spec_baseline", fronts=2, min_n=2, max_n=8)
    assert result["survivors"][0] == "spec_baseline"  # baseline always first
    assert "spec_good" in result["survivors"]
    assert "spec_cheap" in result["survivors"]  # the cheap/low-quality extreme isn't excluded
    assert len(result["survivor_paths"]) == len(result["survivors"])
    # cwd was chdir'd to tmp_path above, so these relative paths resolve directly
    for p in result["survivor_paths"]:
        assert Path(p).exists()
    con.close()
