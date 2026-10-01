import pytest

from harness.gpu import Sample, summarize_samples
from harness.spec import CandidateSpec


def test_spec_hash_stable_regardless_of_dict_construction_order():
    a = CandidateSpec(base_model="m.gguf", quant="q4_k_m", ctx=8192, kv_type_k="q8_0", kv_type_v="q8_0",
                       flash_attn=True)
    b = CandidateSpec(flash_attn=True, kv_type_v="q8_0", kv_type_k="q8_0", ctx=8192, quant="Q4_K_M",
                       base_model="m.gguf")
    assert a.spec_hash == b.spec_hash
    assert a.artifact_hash == b.artifact_hash


def test_spec_hash_path_independent():
    a = CandidateSpec(base_model="models/m.gguf", quant="Q4_K_M")
    b = CandidateSpec(base_model="C:/elsewhere/m.gguf", quant="Q4_K_M")
    assert a.spec_hash == b.spec_hash
    assert a.artifact_hash == b.artifact_hash


def test_spec_hash_changes_with_quant():
    a = CandidateSpec(base_model="m.gguf", quant="Q4_K_M")
    b = CandidateSpec(base_model="m.gguf", quant="Q6_K")
    assert a.spec_hash != b.spec_hash
    assert a.artifact_hash != b.artifact_hash


def test_artifact_hash_ignores_kv_and_ctx_but_spec_hash_does_not():
    # The whole point: two specs that build the SAME weights file (same quant)
    # but measure different KV configs must share one artifact, not rebuild.
    a = CandidateSpec(base_model="m.gguf", quant="Q4_K_M", kv_type_k="f16", kv_type_v="f16", ctx=4096)
    b = CandidateSpec(base_model="m.gguf", quant="Q4_K_M", kv_type_k="q8_0", kv_type_v="q8_0",
                       flash_attn=True, ctx=32768)
    assert a.artifact_hash == b.artifact_hash
    assert a.spec_hash != b.spec_hash


def test_quantized_v_cache_requires_flash_attention():
    with pytest.raises(ValueError):
        CandidateSpec(base_model="m.gguf", quant="Q4_K_M", kv_type_v="q8_0", flash_attn=False)
    # K-only quantization doesn't need flash attention.
    CandidateSpec(base_model="m.gguf", quant="Q4_K_M", kv_type_k="q8_0", flash_attn=False)


def test_unknown_kv_type_rejected():
    with pytest.raises(ValueError):
        CandidateSpec(base_model="m.gguf", quant="Q4_K_M", kv_type_k="not_a_type")


def test_from_json_rejects_unknown_fields(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"base_model": "m.gguf", "quant": "Q4_K_M", "contex": 8192}')
    with pytest.raises(ValueError):
        CandidateSpec.from_json(bad)


def test_from_json_round_trip(tmp_path):
    spec = CandidateSpec(base_model="m.gguf", quant="Q4_K_M", ctx=16384, kv_type_v="q8_0", flash_attn=True)
    path = tmp_path / "spec.json"
    spec.to_json(path)
    loaded = CandidateSpec.from_json(path)
    assert loaded.spec_hash == spec.spec_hash


def test_summarize_samples_empty():
    s = summarize_samples([])
    assert s["n_samples"] == 0
    assert s["peak_vram_mb"] is None


def test_summarize_samples_computes_energy_and_peak_delta():
    samples = [
        Sample(t=0.0, vram_used_mb=2000, power_w=100, energy_mj=1_000_000, temp_c=60, sm_clock_mhz=1800),
        Sample(t=0.5, vram_used_mb=2500, power_w=150, energy_mj=1_050_000, temp_c=62, sm_clock_mhz=1800),
        Sample(t=1.0, vram_used_mb=2300, power_w=120, energy_mj=1_100_000, temp_c=63, sm_clock_mhz=1800),
    ]
    s = summarize_samples(samples, baseline_vram_mb=1500, n_tokens=100)
    assert s["n_samples"] == 3
    assert s["peak_vram_mb"] == 2500
    assert s["peak_vram_delta_mb"] == 1000
    assert s["max_temp_c"] == 63
    assert s["mean_power_w"] == pytest.approx((100 + 150 + 120) / 3)
    assert s["energy_j"] == pytest.approx((1_100_000 - 1_000_000) / 1000)
    assert s["joules_per_token"] == pytest.approx(s["energy_j"] / 100)


def test_summarize_samples_single_sample_has_no_energy():
    s = summarize_samples([Sample(t=0.0, vram_used_mb=2000, power_w=100, energy_mj=1_000_000,
                                   temp_c=60, sm_clock_mhz=1800)])
    assert s["energy_j"] is None
    assert s["joules_per_token"] is None
