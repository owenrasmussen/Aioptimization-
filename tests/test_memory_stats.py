from harness.memory import ModelShape, kv_cache_bytes
from harness.kld import parse
from harness.stats import bootstrap_diff_ci, bootstrap_prop_diff_ci, paired_rel_diff_ci, summarize


def test_kv_cache_llama3_8b_f16():
    # Llama 3 8B: 32 layers, 8 KV heads, head dim 128 -> 128 KiB per token at f16
    shape = ModelShape(kv_layers=32, kv_heads=8, key_dim=128, value_dim=128)
    assert kv_cache_bytes(shape, 1) == 131072
    assert kv_cache_bytes(shape, 32768) == 4 * 2**30  # 4 GiB at 32k


def test_kv_cache_q8_is_about_half_of_f16():
    shape = ModelShape(32, 8, 128, 128)
    ratio = kv_cache_bytes(shape, 1000, "q8_0", "q8_0") / kv_cache_bytes(shape, 1000)
    assert abs(ratio - 34 / 64) < 1e-9


def test_summarize_and_diff_ci():
    a = [100.0, 101.0, 99.5, 100.5, 100.2]
    b = [110.0, 111.0, 109.5, 110.5, 110.2]
    assert summarize(a)["median"] == 100.2
    lo, hi = bootstrap_diff_ci(a, b)
    assert lo > 0  # clearly different
    lo, hi = bootstrap_diff_ci(a, a)
    assert lo <= 0 <= hi


def test_paired_rel_diff_ci_identical_series_contains_zero():
    a = [100.0, 101.0, 99.5, 100.5, 100.2, 99.8]
    lo, hi = paired_rel_diff_ci(a, a)
    assert lo <= 0 <= hi


def test_paired_rel_diff_ci_detects_real_relative_shift():
    a = [100.0, 101.0, 99.5, 100.5, 100.2, 99.8]
    b = [x * 1.10 for x in a]  # exactly 10% faster every round, paired
    lo, hi = paired_rel_diff_ci(a, b)
    assert lo > 0.05  # clearly excludes 0, centered near +0.10


def test_paired_rel_diff_ci_requires_equal_length():
    import pytest
    with pytest.raises(ValueError):
        paired_rel_diff_ci([1.0, 2.0], [1.0])


def test_bootstrap_prop_diff_ci_identical_proportions_contains_zero():
    lo, hi = bootstrap_prop_diff_ci(k_a=20, n_a=30, k_b=20, n_b=30)
    assert lo <= 0 <= hi


def test_bootstrap_prop_diff_ci_detects_real_difference():
    # 90% vs 30% on reasonably sized samples should clearly exclude 0
    lo, hi = bootstrap_prop_diff_ci(k_a=9, n_a=30, k_b=27, n_b=30)
    assert lo > 0


def test_parse_kld_log():
    log = """
Mean PPL(Q)                   :  10.123456 ±   0.1
Mean    KLD:   0.012345 ±   0.000123
Maximum KLD:   3.456789
99.0%   KLD:   0.234567
Same top p: 95.123 ± 0.123 %
"""
    r = parse(log)
    assert r == {"ppl": 10.123456, "kld_mean": 0.012345, "kld_p99": 0.234567,
                 "kld_max": 3.456789, "top1_agree_pct": 95.123}
