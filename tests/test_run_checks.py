"""Regression coverage for the Week 4 red-team checks in harness/run.py:
_check_bench_record (bench-record validation) and the physics check wired
into measure_bench. These need a GPU-model-shaped fake bench record but no
actual GPU or server -- llama.bench is mocked.
"""
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from harness import physics
from harness import run as run_mod
from harness.spec import CandidateSpec

MODEL_PATH = Path("models/qwen2.5-3b-instruct-Q4_K_M.gguf")
requires_model = pytest.mark.skipif(not MODEL_PATH.exists(), reason="model file not present in this environment")

_SPEC = CandidateSpec(base_model="models/qwen2.5-3b-instruct-f16.gguf", quant="Q4_K_M", ctx=8192,
                       flash_attn=True)


def _good_record(**overrides) -> dict:
    rec = {"n_prompt": 0, "n_gen": 64, "n_depth": 8192, "type_k": "f16", "type_v": "f16",
           "flash_attn": 1, "n_gpu_layers": 99, "model_filename": str(MODEL_PATH),
           "avg_ts": 150.0, "model_n_params": 3_000_000_000, "model_size": 2_000_000_000}
    rec.update(overrides)
    return rec


def test_check_bench_record_clean_record_has_no_problems():
    problems = run_mod._check_bench_record([_good_record()], _SPEC, MODEL_PATH, pp=0, tg=64, depth=8192)
    assert problems == []


def test_check_bench_record_catches_multiple_records():
    # depth-smuggling via engine_flags could cause llama-bench to return
    # more than one record (R5 in the Week 4 red-team exercise)
    problems = run_mod._check_bench_record([_good_record(), _good_record()], _SPEC, MODEL_PATH,
                                            pp=0, tg=64, depth=8192)
    assert len(problems) == 1
    assert "1 bench record" in problems[0]


def test_check_bench_record_catches_wrong_depth():
    problems = run_mod._check_bench_record([_good_record(n_depth=0)], _SPEC, MODEL_PATH,
                                            pp=0, tg=64, depth=8192)
    assert any("n_depth" in p for p in problems)


def test_check_bench_record_catches_kv_type_mismatch():
    # The exact shape of the original KL-missing-KV-flags-style bug, now
    # checked at the bench level too: a record claiming a different KV
    # type than the spec declared.
    problems = run_mod._check_bench_record([_good_record(type_k="q4_0")], _SPEC, MODEL_PATH,
                                            pp=0, tg=64, depth=8192)
    assert any("type_k" in p for p in problems)


def test_check_bench_record_catches_wrong_model_file():
    problems = run_mod._check_bench_record([_good_record(model_filename="models/some_other_model.gguf")],
                                            _SPEC, MODEL_PATH, pp=0, tg=64, depth=8192)
    assert any("model_filename" in p for p in problems)


@requires_model
def test_measure_bench_marks_invalid_on_bench_record_mismatch():
    run_dir = Path(tempfile.mkdtemp())
    bandwidth = {"gbps": None, "peak_tops": None}  # skip physics, isolate the record check
    with mock.patch("harness.llama.bench", return_value=[_good_record(n_depth=0)]):  # wrong depth
        entry = run_mod.measure_bench(_SPEC, MODEL_PATH, "bench_tg", 0, 64, 0, [], run_dir, 90,
                                       "third_party/llama.cpp/build/bin", 0.8, bandwidth)
    assert entry["status"] == "invalid"
    assert entry["failure"]["type"] == "bench_mismatch"


@requires_model
def test_measure_bench_marks_invalid_on_impossible_speed():
    # R7: an inflated/impossible tg_tps must fail the physics check, not be
    # silently recorded as a clean result.
    run_dir = Path(tempfile.mkdtemp())
    bandwidth = physics.peak_bandwidth_gbps()
    bandwidth["peak_tops"] = physics.KNOWN_PEAK_DENSE_TOPS.get("NVIDIA GeForce RTX 3080 Ti")
    with mock.patch("harness.llama.bench", return_value=[_good_record(avg_ts=50_000.0)]):
        entry = run_mod.measure_bench(_SPEC, MODEL_PATH, "bench_tg", 0, 64, 0, [], run_dir, 90,
                                       "third_party/llama.cpp/build/bin", 0.8, bandwidth)
    assert entry["status"] == "invalid"
    assert entry["failure"]["type"] == "physics"
    assert entry["checks"][0]["passed"] is False
    assert entry["checks"][0]["details"]["utilization"] > 1.0


@requires_model
def test_ensure_artifact_computes_n_params_and_bpw_from_gguf():
    # Week 5: a --no-bench gate stage never runs llama-bench, so n_params/bpw
    # must come from the GGUF directly, not be left to a bench record that
    # might never exist.
    artifact = run_mod.ensure_artifact(_SPEC, MODEL_PATH.parent / "cache", Path(tempfile.mkdtemp()),
                                        "third_party/llama.cpp/build/bin")
    assert artifact["n_params"] is not None
    assert 3_000_000_000 < artifact["n_params"] < 4_000_000_000  # Qwen2.5-3B
    assert 4.5 < artifact["bpw"] < 5.5  # Q4_K_M is ~4.9-4.95 bpw on this model


def test_ensure_artifact_raises_on_quant_missing_from_file_type_map(tmp_path):
    # Week 5 fix: a quant missing from _QUANT_FILE_TYPE must fail loudly, not
    # silently skip the Week 4 cache-poisoning check for exactly the quants
    # nobody's added an entry for yet.
    spec = CandidateSpec(base_model="models/qwen2.5-3b-instruct-f16.gguf", quant="IQ4_XS", ctx=8192)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / f"{spec.artifact_hash}.gguf").write_bytes(b"not a real gguf, just needs to exist")
    with pytest.raises(ValueError, match="no entry in _QUANT_FILE_TYPE"):
        run_mod.ensure_artifact(spec, cache_dir, tmp_path, "third_party/llama.cpp/build/bin")


@requires_model
def test_measure_bench_passes_honest_speed():
    run_dir = Path(tempfile.mkdtemp())
    bandwidth = physics.peak_bandwidth_gbps()
    bandwidth["peak_tops"] = physics.KNOWN_PEAK_DENSE_TOPS.get("NVIDIA GeForce RTX 3080 Ti")
    with mock.patch("harness.llama.bench", return_value=[_good_record(avg_ts=150.0)]):
        entry = run_mod.measure_bench(_SPEC, MODEL_PATH, "bench_tg", 0, 64, 0, [], run_dir, 90,
                                       "third_party/llama.cpp/build/bin", 0.8, bandwidth)
    assert entry["status"] == "ok"
    assert entry["checks"][0]["passed"] is True
