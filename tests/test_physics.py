from pathlib import Path

import pytest

from harness import physics

Q4KM_PATH = Path("models/qwen2.5-3b-instruct-Q4_K_M.gguf")
requires_model = pytest.mark.skipif(not Q4KM_PATH.exists(), reason="model file not present in this environment")


def test_check_decode_passes_within_bound():
    # ~10 GB/s worth of headroom under a 900 GB/s peak with 2GB bytes/token
    result = physics.check_decode(tg_tps=200, bytes_per_token=2_000_000_000, peak_gbps=900)
    assert result["passed"] is True
    assert result["bound"] == pytest.approx(900e9 / 2_000_000_000)


def test_check_decode_fails_impossible_speed():
    # Same bytes/token and bandwidth, but claiming 10x the physical bound
    result = physics.check_decode(tg_tps=5000, bytes_per_token=2_000_000_000, peak_gbps=900)
    assert result["passed"] is False
    assert result["value"] == 5000


def test_check_decode_reports_utilization():
    result = physics.check_decode(tg_tps=200, bytes_per_token=2_000_000_000, peak_gbps=900)
    assert 0 < result["details"]["utilization"] < 1


def test_check_decode_none_bound_when_missing_inputs():
    result = physics.check_decode(tg_tps=200, bytes_per_token=None, peak_gbps=900)
    assert result["bound"] is None
    assert result["passed"] is True  # can't check without the inputs -- not a false failure


def test_check_prefill_passes_within_bound():
    # 272.8 TOPS / (2 * 3e9 params) ~= 45,467 tok/s ceiling
    result = physics.check_prefill(pp_tps=8000, n_params=3_000_000_000, peak_tops=272.8)
    assert result["passed"] is True


def test_check_prefill_fails_impossible_speed():
    result = physics.check_prefill(pp_tps=1_000_000, n_params=3_000_000_000, peak_tops=272.8)
    assert result["passed"] is False


@requires_model
def test_decode_bytes_per_token_excludes_token_embd():
    result = physics.decode_bytes_per_token(Q4KM_PATH, depth=0, kv_type_k="f16", kv_type_v="f16")
    assert result is not None
    assert result["kv_bytes"] == 0  # depth=0, no cached tokens yet
    # token_embd.weight (~175MB raw, smaller quantized) must not be counted --
    # verified live that it's ~8.3% of total weight bytes in this GGUF, distinct
    # from output.weight (the LM head, which IS read every decode step).
    from gguf import GGUFReader
    total_including_embd = sum(t.n_bytes for t in GGUFReader(str(Q4KM_PATH)).tensors)
    assert result["weight_bytes"] < total_including_embd


@requires_model
def test_decode_bytes_per_token_grows_with_depth():
    shallow = physics.decode_bytes_per_token(Q4KM_PATH, depth=1000, kv_type_k="f16", kv_type_v="f16")
    deep = physics.decode_bytes_per_token(Q4KM_PATH, depth=32000, kv_type_k="f16", kv_type_v="f16")
    assert deep["kv_bytes"] > shallow["kv_bytes"]
    assert deep["weight_bytes"] == shallow["weight_bytes"]  # weights don't depend on depth


def test_peak_bandwidth_override():
    result = physics.peak_bandwidth_gbps(override=500.0)
    assert result == {"gbps": 500.0, "source": "override", "nvml_gbps": None, "table_gbps": None}
