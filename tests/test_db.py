import json

from harness import db


def _sample_record(run_id="run1", status="ok"):
    return {
        "spec_hash": "spec0000000000ab",
        "artifact_hash": "art00000000000ab",
        "env_id": "env0000000000000",
        "spec": {
            "base_model": "qwen2.5-3b-instruct-f16.gguf", "quant": "Q4_K_M",
            "tensor_overrides": {}, "imatrix": None, "kv_type_k": "q8_0", "kv_type_v": "q8_0",
            "flash_attn": True, "ctx": 8192, "batch_size": None, "ubatch_size": None,
            "engine_flags": [],
        },
        "artifact": {"path": "models/cache/art00000000000ab.gguf", "size_bytes": 2_000_000_000,
                     "n_params": 3_000_000_000, "bpw": 4.94, "build_seconds": 12.3,
                     "build_cmd": "llama-quantize ...", "built_at": "2026-10-01T00:00:00"},
        "environment": {"gpu": "NVIDIA GeForce RTX 3080 Ti", "vram_mb": 12288, "driver": "596.36",
                        "cuda": "13.4", "engine_commit": "f7b384c1e", "power_limit_w": 400.0,
                        "locked_clock_mhz": None},
        "runs": [
            {
                "run_id": run_id, "workload_id": "wl0000000000000a",
                "workload": {"kind": "bench_pp", "input_len": 512, "output_len": 0, "concurrency": 1},
                "repeat": 0, "seed": None, "start_temp_c": 60, "start_sm_clock_mhz": 1800,
                "started_at": "2026-10-01T00:00:01", "wall_seconds": 2.1, "status": status,
                "command": "llama-bench ...", "log_path": "results/runs/x/pp0.log",
                "speed": {"pp_tps": 8000.0, "tg_tps": None, "raw": {"n_prompt": 512}},
                "resources": {"peak_vram_mb": 3500.0, "peak_vram_delta_mb": 2000.0,
                              "predicted_total_mb": 2970.0, "mean_power_w": 140.0, "energy_j": 50.0,
                              "joules_per_token": None, "max_temp_c": 66, "n_samples": 20,
                              "nvml_log": "results/runs/x/pp0_nvml.csv"},
                "quality": None,
                "failure": None,
            },
        ],
    }


def test_schema_creates_cleanly(tmp_path):
    con = db.connect(str(tmp_path / "test.duckdb"))
    tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    assert tables == {"specs", "artifacts", "environments", "workloads", "runs", "speed",
                       "resources", "quality", "failures", "split_files", "schedule", "checks"}
    con.close()


def test_quality_row_with_week3_columns(tmp_path):
    con = db.connect(str(tmp_path / "test.duckdb"))
    record = _sample_record()
    record["runs"][0]["quality"] = {
        "eval_name": "gsm8k", "split": "dev", "score": 0.42, "n_items": 50, "stderr": 0.07,
        "details": {"metric": "exact_match,flexible-extract"},
    }
    db.insert_record(con, record)
    row = con.execute(
        "SELECT score, n_items, stderr, details FROM quality WHERE run_id='run1' AND eval_name='gsm8k'"
    ).fetchone()
    assert row[0] == 0.42
    assert row[1] == 50
    assert row[2] == 0.07
    assert json.loads(row[3])["metric"] == "exact_match,flexible-extract"
    con.close()


def test_sync_split_files_mirrors_manifest(tmp_path):
    manifest_path = tmp_path / "splits.json"
    manifest_path.write_text(json.dumps({
        "version": 1,
        "text": {
            "dev": {"path": "data/dev.txt", "sha256": "abc123"},
            "held_out": {"path": "data/held_out.txt", "sha256": "def456", "locked_at": "2026-10-01T00:00:00"},
        },
        "tasks": {},
    }))
    con = db.connect(str(tmp_path / "test.duckdb"))
    db.sync_split_files(con, manifest_path)
    rows = {r[0]: r[1:] for r in con.execute("SELECT name, path, sha256 FROM split_files").fetchall()}
    assert rows["dev"] == ("data/dev.txt", "abc123")
    assert rows["held_out"] == ("data/held_out.txt", "def456")
    con.close()


def test_split_files_survives_rebuild(tmp_path):
    # The whole point: rebuild() deletes the .duckdb file, but the manifest
    # lives in a separate file (data/splits.json in production), so the
    # held-out lock isn't lost -- the next sync_split_files() call restores it.
    manifest_path = tmp_path / "splits.json"
    manifest_path.write_text(json.dumps({"version": 1, "text": {
        "held_out": {"path": "x", "sha256": "lockedhash", "locked_at": "2026-10-01T00:00:00"}}, "tasks": {}}))

    db_path = tmp_path / "runs.duckdb"
    con = db.connect(str(db_path))
    db.sync_split_files(con, manifest_path)
    assert con.execute("SELECT sha256 FROM split_files WHERE name='held_out'").fetchone()[0] == "lockedhash"
    con.close()

    db.rebuild(str(db_path), str(tmp_path / "runs"))  # deletes and recreates db_path
    # connect() always auto-syncs from the real project data/splits.json when no
    # manifest_path is given (that's the whole point -- a fresh connect() in
    # production restores the real lock), so this test re-syncs its own tmp
    # manifest explicitly rather than asserting on connect()'s implicit default,
    # which would otherwise pick up whatever this repo's actual data/splits.json
    # happens to contain.
    con = db.connect(str(db_path))
    db.sync_split_files(con, manifest_path)
    assert con.execute("SELECT sha256 FROM split_files WHERE name='held_out'").fetchone()[0] == "lockedhash"
    con.close()


def test_insert_record_round_trip(tmp_path):
    con = db.connect(str(tmp_path / "test.duckdb"))
    db.insert_record(con, _sample_record())

    assert con.execute("SELECT count(*) FROM specs").fetchone()[0] == 1
    assert con.execute("SELECT count(*) FROM artifacts").fetchone()[0] == 1
    assert con.execute("SELECT count(*) FROM environments").fetchone()[0] == 1
    assert con.execute("SELECT count(*) FROM workloads").fetchone()[0] == 1
    assert con.execute("SELECT count(*) FROM runs").fetchone()[0] == 1
    assert con.execute("SELECT pp_tps FROM speed WHERE run_id='run1'").fetchone()[0] == 8000.0
    assert con.execute("SELECT peak_vram_delta_mb FROM resources WHERE run_id='run1'").fetchone()[0] == 2000.0
    con.close()


def test_insert_record_twice_does_not_duplicate_spec_env_or_workload(tmp_path):
    con = db.connect(str(tmp_path / "test.duckdb"))
    record = _sample_record()
    db.insert_record(con, record)
    db.insert_record(con, record)  # same record again, e.g. a rebuild replay

    assert con.execute("SELECT count(*) FROM specs").fetchone()[0] == 1
    assert con.execute("SELECT count(*) FROM environments").fetchone()[0] == 1
    assert con.execute("SELECT count(*) FROM workloads").fetchone()[0] == 1
    assert con.execute("SELECT count(*) FROM runs").fetchone()[0] == 1
    con.close()


def test_two_specs_share_one_artifact_row(tmp_path):
    # The whole point of the two-hash scheme: different spec_hash, same artifact_hash.
    con = db.connect(str(tmp_path / "test.duckdb"))
    r1 = _sample_record(run_id="run1")
    r2 = _sample_record(run_id="run2")
    r2["spec_hash"] = "spec_different_ctx"
    r2["spec"]["ctx"] = 32768
    r2["runs"][0]["run_id"] = "run2"
    r2["runs"][0]["workload_id"] = "wl_different"
    db.insert_record(con, r1)
    db.insert_record(con, r2)

    assert con.execute("SELECT count(*) FROM specs").fetchone()[0] == 2
    assert con.execute("SELECT count(*) FROM artifacts").fetchone()[0] == 1
    assert con.execute("SELECT count(*) FROM runs").fetchone()[0] == 2
    con.close()


def test_failure_is_recorded(tmp_path):
    con = db.connect(str(tmp_path / "test.duckdb"))
    record = _sample_record(status="failed")
    record["runs"][0]["speed"] = None
    record["runs"][0]["resources"] = None
    record["runs"][0]["failure"] = {"type": "bench", "message": "out of memory", "log_path": "x.log"}
    db.insert_record(con, record)

    row = con.execute("SELECT type, message FROM failures WHERE run_id='run1'").fetchone()
    assert row == ("bench", "out of memory")
    con.close()


def test_schedule_and_checks_recorded(tmp_path):
    con = db.connect(str(tmp_path / "test.duckdb"))
    record = _sample_record()
    record["runs"][0]["schedule"] = {"session_id": "sess1", "arm": 0, "round": 3, "order_pos": 1}
    record["runs"][0]["checks"] = [
        {"name": "physics_decode", "passed": True, "value": 150.0, "bound": 400.0,
         "details": {"utilization": 0.375}},
    ]
    db.insert_record(con, record)

    sched = con.execute("SELECT session_id, arm, round, order_pos FROM schedule WHERE run_id='run1'").fetchone()
    assert sched == ("sess1", 0, 3, 1)
    check = con.execute("SELECT name, passed, value, bound FROM checks WHERE run_id='run1'").fetchone()
    assert check == ("physics_decode", True, 150.0, 400.0)
    con.close()


def test_rebuild_replays_json_files(tmp_path):
    runs_dir = tmp_path / "results" / "runs" / "run1"
    runs_dir.mkdir(parents=True)
    (runs_dir / "record.json").write_text(json.dumps(_sample_record()))
    db_path = tmp_path / "results" / "runs.duckdb"

    n = db.rebuild(str(db_path), str(tmp_path / "results" / "runs"))
    assert n == 1

    con = db.connect(str(db_path))
    assert con.execute("SELECT count(*) FROM runs").fetchone()[0] == 1
    con.close()
