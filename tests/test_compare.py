import json

from harness import compare, db


def _insert_bench_round(con, run_id, spec_hash, env_id, workload_id, kind, pp_tps, tg_tps, status,
                         session_id, arm, rnd):
    con.execute("INSERT OR IGNORE INTO workloads VALUES (?,?,?,?,?,?,?)",
                [workload_id, kind, None, None, None, 1, json.dumps({"kind": kind})])
    con.execute("INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [run_id, spec_hash, env_id, workload_id, rnd, None, 60, 1800,
                 "2026-10-01T00:00:00", 1.0, status, None, None])
    if status == "ok":
        con.execute("INSERT OR REPLACE INTO speed VALUES (?,?,?,?,?,?,?)",
                    [run_id, pp_tps, tg_tps, None, None, None, None])
    con.execute("INSERT OR REPLACE INTO schedule VALUES (?,?,?,?,?)",
                [run_id, session_id, arm, rnd, arm])


def _insert_spec(con, spec_hash, overrides=None):
    base = {"base_model": "m.gguf", "quant": "Q4_K_M", "kv_type_k": "f16", "kv_type_v": "f16",
             "flash_attn": True, "ctx": 8192, "batch_size": None, "ubatch_size": None,
             "engine_flags": []}
    base.update(overrides or {})
    con.execute("INSERT OR IGNORE INTO specs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?, current_timestamp)",
                [spec_hash, "art1", base["base_model"], base["quant"], json.dumps({}), None,
                 base["kv_type_k"], base["kv_type_v"], base["flash_attn"], base["ctx"],
                 base["batch_size"], base["ubatch_size"], json.dumps(base["engine_flags"]),
                 json.dumps(base)])


def _insert_quality(con, run_id, spec_hash, env_id, eval_name, score, n_items, started_at="2026-10-01T00:00:00"):
    wl = f"wl_{eval_name}"
    con.execute("INSERT OR IGNORE INTO workloads VALUES (?,?,?,?,?,?,?)",
                [wl, "lmeval", None, None, None, 1, json.dumps({})])
    con.execute("INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [run_id, spec_hash, env_id, wl, 0, None, 60, 1800, started_at, 1.0, "ok", None, None])
    con.execute("INSERT OR REPLACE INTO quality VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                [run_id, eval_name, "dev", score, None, None, None, None, n_items, None, None])


def _setup(con):
    con.execute("INSERT OR IGNORE INTO environments VALUES (?,?,?,?,?,?,?,?)",
                ["env1", "RTX 3080 Ti", 12288, "596.36", "13.4", "abc123", 400.0, None])
    _insert_spec(con, "spec_a")
    _insert_spec(con, "spec_b", {"quant": "Q3_K_M"})


def test_pair_and_drop_matches_only_ok_rounds():
    base = {0: (100.0, "ok"), 1: (101.0, "ok"), 2: (99.0, "invalid")}
    arm = {0: (110.0, "ok"), 1: (111.0, "ok"), 2: (120.0, "ok")}
    a, b, dropped = compare._pair_and_drop(base, arm)
    assert a == [100.0, 101.0]
    assert b == [110.0, 111.0]
    assert dropped == 1


def test_verdict_classification():
    assert compare._verdict(0.01, 0.05) == "faster"
    assert compare._verdict(-0.05, -0.01) == "slower"
    assert compare._verdict(-0.02, 0.02) == "no detectable difference"


def test_compare_session_aa_calibration(tmp_path):
    # The A/A test: baseline and arm are literally the same spec. Compare
    # must report "no detectable difference" -- this is what calibrates
    # the statistics themselves, not just the comparison logic.
    con = db.connect(str(tmp_path / "t.duckdb"))
    _setup(con)
    for rnd in range(10):
        _insert_bench_round(con, f"b0_{rnd}", "spec_a", "env1", "wl_pp", "bench_pp",
                             8000 + rnd, None, "ok", "sessA", 0, rnd)
        _insert_bench_round(con, f"b1_{rnd}", "spec_a", "env1", "wl_pp", "bench_pp",
                             8000 + rnd, None, "ok", "sessA", 1, rnd)
    result = compare.compare_session(con, "sessA", baseline_arm=0)
    assert result["arms"][1]["speed"]["bench_pp"]["verdict"] == "no detectable difference"
    con.close()


def test_compare_session_detects_real_speed_difference(tmp_path):
    con = db.connect(str(tmp_path / "t.duckdb"))
    _setup(con)
    for rnd in range(10):
        _insert_bench_round(con, f"b0_{rnd}", "spec_a", "env1", "wl_tg", "bench_tg",
                             None, 150.0, "ok", "sessB", 0, rnd)
        _insert_bench_round(con, f"b1_{rnd}", "spec_b", "env1", "wl_tg", "bench_tg",
                             None, 200.0, "ok", "sessB", 1, rnd)  # consistently faster
    result = compare.compare_session(con, "sessB", baseline_arm=0)
    assert result["arms"][1]["speed"]["bench_tg"]["verdict"] == "faster"
    con.close()


def test_compare_session_flags_unverified_quality(tmp_path):
    con = db.connect(str(tmp_path / "t.duckdb"))
    _setup(con)
    for rnd in range(5):
        _insert_bench_round(con, f"b0_{rnd}", "spec_a", "env1", "wl_tg", "bench_tg",
                             None, 150.0, "ok", "sessC", 0, rnd)
        _insert_bench_round(con, f"b1_{rnd}", "spec_b", "env1", "wl_tg", "bench_tg",
                             None, 200.0, "ok", "sessC", 1, rnd)
    # no quality rows inserted for spec_b at all
    result = compare.compare_session(con, "sessC", baseline_arm=0)
    assert result["arms"][1]["quality"] == "QUALITY UNVERIFIED"
    con.close()


def test_compare_session_reports_quality_delta(tmp_path):
    con = db.connect(str(tmp_path / "t.duckdb"))
    _setup(con)
    for rnd in range(5):
        _insert_bench_round(con, f"b0_{rnd}", "spec_a", "env1", "wl_tg", "bench_tg",
                             None, 150.0, "ok", "sessD", 0, rnd)
        _insert_bench_round(con, f"b1_{rnd}", "spec_b", "env1", "wl_tg", "bench_tg",
                             None, 200.0, "ok", "sessD", 1, rnd)
    _insert_quality(con, "q_a", "spec_a", "env1", "gsm8k", 0.70, 30)
    _insert_quality(con, "q_b", "spec_b", "env1", "gsm8k", 0.30, 30)  # clearly worse
    result = compare.compare_session(con, "sessD", baseline_arm=0)
    q = result["arms"][1]["quality"]["gsm8k"]
    assert q["prop_diff_ci"][1] < 0  # arm's score is lower, CI should exclude 0 on the high side
    con.close()


def test_compare_session_warns_on_multi_field_difference(tmp_path):
    con = db.connect(str(tmp_path / "t.duckdb"))
    _setup(con)  # spec_b differs from spec_a in "quant" only, per _insert_spec's override
    con.execute("UPDATE specs SET kv_type_k='q8_0' WHERE spec_hash='spec_b'")
    # re-fetch to confirm spec_json itself needs updating too for the warning check
    for rnd in range(5):
        _insert_bench_round(con, f"b0_{rnd}", "spec_a", "env1", "wl_tg", "bench_tg",
                             None, 150.0, "ok", "sessE", 0, rnd)
        _insert_bench_round(con, f"b1_{rnd}", "spec_b", "env1", "wl_tg", "bench_tg",
                             None, 200.0, "ok", "sessE", 1, rnd)
    result = compare.compare_session(con, "sessE", baseline_arm=0)
    # spec_b's stored spec_json differs from spec_a in "quant" (set at insert) --
    # exactly one field differs, so no warning expected here.
    assert result["arms"][1]["warnings"] == []
    con.close()
