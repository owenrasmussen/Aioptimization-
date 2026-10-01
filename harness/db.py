"""DuckDB schema and insert helpers.

Design note: no foreign keys. DuckDB's FK support makes the INSERT OR IGNORE /
INSERT OR REPLACE idempotency this module relies on awkward, and nothing here
needs enforced referential integrity -- joins work fine without it.

Each JSON file under results/runs/ (written by harness/run.py) is the source
of truth; this database is a queryable, rebuildable summary of those files.
If the schema changes, `python -m harness.db --rebuild` drops every table and
re-ingests all the JSON, rather than trying to migrate rows in place.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb

DB_PATH = "results/runs.duckdb"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS specs (
  spec_hash        VARCHAR PRIMARY KEY,
  artifact_hash    VARCHAR NOT NULL,
  base_model       VARCHAR NOT NULL,
  quant            VARCHAR NOT NULL,
  tensor_overrides JSON,
  imatrix          VARCHAR,
  kv_type_k        VARCHAR NOT NULL,
  kv_type_v        VARCHAR NOT NULL,
  flash_attn       BOOLEAN NOT NULL,
  ctx              INTEGER NOT NULL,
  batch_size       INTEGER,
  ubatch_size      INTEGER,
  engine_flags     JSON,
  spec_json        JSON NOT NULL,
  created_at       TIMESTAMP DEFAULT current_timestamp
);

CREATE TABLE IF NOT EXISTS artifacts (
  artifact_hash VARCHAR PRIMARY KEY,
  path          VARCHAR NOT NULL,
  size_bytes    BIGINT NOT NULL,
  n_params      BIGINT,
  bpw           DOUBLE,
  build_seconds DOUBLE,
  build_cmd     VARCHAR,
  built_at      TIMESTAMP
);

CREATE TABLE IF NOT EXISTS environments (
  env_id           VARCHAR PRIMARY KEY,
  gpu              VARCHAR NOT NULL,
  vram_mb          INTEGER,
  driver           VARCHAR,
  cuda             VARCHAR,
  engine_commit    VARCHAR NOT NULL,
  power_limit_w    DOUBLE,
  locked_clock_mhz INTEGER
);

CREATE TABLE IF NOT EXISTS workloads (
  workload_id     VARCHAR PRIMARY KEY,
  kind            VARCHAR NOT NULL,
  prompt_set_hash VARCHAR,
  input_len       INTEGER,
  output_len      INTEGER,
  concurrency     INTEGER DEFAULT 1,
  params          JSON NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
  run_id             VARCHAR PRIMARY KEY,
  spec_hash          VARCHAR NOT NULL,
  env_id             VARCHAR NOT NULL,
  workload_id        VARCHAR NOT NULL,
  repeat             INTEGER NOT NULL,
  seed               INTEGER,
  start_temp_c       INTEGER,
  start_sm_clock_mhz INTEGER,
  started_at         TIMESTAMP NOT NULL,
  wall_seconds       DOUBLE,
  status             VARCHAR NOT NULL,
  command            VARCHAR,
  log_path           VARCHAR
);

CREATE TABLE IF NOT EXISTS speed (
  run_id  VARCHAR PRIMARY KEY,
  pp_tps  DOUBLE,
  tg_tps  DOUBLE,
  ttft_ms DOUBLE,
  p50_ms  DOUBLE,
  p95_ms  DOUBLE,
  raw     JSON
);

CREATE TABLE IF NOT EXISTS resources (
  run_id             VARCHAR PRIMARY KEY,
  peak_vram_mb       DOUBLE,
  peak_vram_delta_mb DOUBLE,
  predicted_total_mb DOUBLE,
  mean_power_w       DOUBLE,
  energy_j           DOUBLE,
  joules_per_token   DOUBLE,
  max_temp_c         INTEGER,
  n_samples          INTEGER,
  nvml_log           VARCHAR
);

CREATE TABLE IF NOT EXISTS quality (
  run_id         VARCHAR NOT NULL,
  eval_name      VARCHAR NOT NULL,
  split          VARCHAR NOT NULL,
  score          DOUBLE,
  kl_mean        DOUBLE,
  kl_p99         DOUBLE,
  kl_max         DOUBLE,
  top1_agree_pct DOUBLE,
  PRIMARY KEY (run_id, eval_name)
);

CREATE TABLE IF NOT EXISTS failures (
  run_id   VARCHAR NOT NULL,
  type     VARCHAR NOT NULL,
  message  VARCHAR,
  log_path VARCHAR,
  PRIMARY KEY (run_id, type)
);
"""


def connect(path: str = DB_PATH) -> duckdb.DuckDBPyConnection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(path)
    con.execute(_SCHEMA)
    return con


def insert_record(con: duckdb.DuckDBPyConnection, record: dict) -> None:
    """Insert one result record (the dict harness/run.py writes to
    results/runs/<id>/record.json). Idempotent: calling this twice with the
    same record does not duplicate rows, so db.rebuild() can just replay
    every JSON file in order."""
    spec = record["spec"]
    con.execute(
        "INSERT OR IGNORE INTO specs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?, current_timestamp)",
        [record["spec_hash"], record["artifact_hash"], spec["base_model"], spec["quant"],
         json.dumps(spec["tensor_overrides"]), spec["imatrix"], spec["kv_type_k"], spec["kv_type_v"],
         spec["flash_attn"], spec["ctx"], spec["batch_size"], spec["ubatch_size"],
         json.dumps(list(spec["engine_flags"])), json.dumps(spec)],
    )

    a = record["artifact"]
    con.execute(
        "INSERT OR IGNORE INTO artifacts VALUES (?,?,?,?,?,?,?,?)",
        [record["artifact_hash"], a["path"], a["size_bytes"], a.get("n_params"), a.get("bpw"),
         a.get("build_seconds"), a.get("build_cmd"), a.get("built_at")],
    )

    e = record["environment"]
    con.execute(
        "INSERT OR IGNORE INTO environments VALUES (?,?,?,?,?,?,?,?)",
        [record["env_id"], e["gpu"], e.get("vram_mb"), e.get("driver"), e.get("cuda"),
         e["engine_commit"], e.get("power_limit_w"), e.get("locked_clock_mhz")],
    )

    for r in record["runs"]:
        w = r["workload"]
        con.execute(
            "INSERT OR IGNORE INTO workloads VALUES (?,?,?,?,?,?,?)",
            [r["workload_id"], w["kind"], w.get("prompt_set_hash"), w.get("input_len"),
             w.get("output_len"), w.get("concurrency", 1), json.dumps(w)],
        )
        con.execute(
            "INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [r["run_id"], record["spec_hash"], record["env_id"], r["workload_id"], r["repeat"],
             r.get("seed"), r.get("start_temp_c"), r.get("start_sm_clock_mhz"), r["started_at"],
             r.get("wall_seconds"), r["status"], r.get("command"), r.get("log_path")],
        )
        if r.get("speed"):
            s = r["speed"]
            con.execute(
                "INSERT OR REPLACE INTO speed VALUES (?,?,?,?,?,?,?)",
                [r["run_id"], s.get("pp_tps"), s.get("tg_tps"), s.get("ttft_ms"),
                 s.get("p50_ms"), s.get("p95_ms"), json.dumps(s.get("raw"))],
            )
        if r.get("resources"):
            res = r["resources"]
            con.execute(
                "INSERT OR REPLACE INTO resources VALUES (?,?,?,?,?,?,?,?,?,?)",
                [r["run_id"], res.get("peak_vram_mb"), res.get("peak_vram_delta_mb"),
                 res.get("predicted_total_mb"), res.get("mean_power_w"), res.get("energy_j"),
                 res.get("joules_per_token"), res.get("max_temp_c"), res.get("n_samples"),
                 res.get("nvml_log")],
            )
        if r.get("quality"):
            q = r["quality"]
            con.execute(
                "INSERT OR REPLACE INTO quality VALUES (?,?,?,?,?,?,?,?)",
                [r["run_id"], q.get("eval_name", "kld"), q.get("split", "dev"), q.get("score"),
                 q.get("kl_mean"), q.get("kl_p99"), q.get("kl_max"), q.get("top1_agree_pct")],
            )
        if r.get("failure"):
            f = r["failure"]
            con.execute(
                "INSERT OR REPLACE INTO failures VALUES (?,?,?,?)",
                [r["run_id"], f["type"], f.get("message"), f.get("log_path")],
            )


def rebuild(path: str = DB_PATH, runs_dir: str = "results/runs") -> int:
    """Drop and recreate every table, then replay every results/runs/*/record.json.
    The full migration story for a schema change: fix insert_record/the schema,
    then rerun this -- no in-place ALTER TABLE bookkeeping to maintain."""
    db_file = Path(path)
    if db_file.exists():
        db_file.unlink()
    con = connect(path)
    n = 0
    for record_path in sorted(Path(runs_dir).glob("*/record.json")):
        insert_record(con, json.loads(record_path.read_text()))
        n += 1
    con.close()
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="drop and re-ingest all results/runs/*/record.json")
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--runs-dir", default="results/runs")
    args = ap.parse_args()
    if args.rebuild:
        n = rebuild(args.db, args.runs_dir)
        print(f"rebuilt {args.db} from {n} record(s)")
    else:
        connect(args.db).close()
        print(f"schema ensured at {args.db}")


if __name__ == "__main__":
    main()
