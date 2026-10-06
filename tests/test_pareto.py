from harness import pareto


def _row(spec_hash, quant, kv, predicted_mb, tg_tps, short_kl, long_kl):
    return {
        "spec_hash": spec_hash, "quant": quant, "kv_type_k": kv, "kv_type_v": kv,
        "ctx": 32768, "bpw": 4.9, "predicted_total_mb": predicted_mb,
        "measured_vram_delta_mb": predicted_mb * 0.9, "pp_tps": tg_tps * 20, "tg_tps": tg_tps,
        "quality": {
            "kld_c512_n20": {"split": "search", "score": None, "kl_mean": short_kl, "n_items": None},
            "kld_c16384_n2": {"split": "search", "score": None, "kl_mean": long_kl, "n_items": None},
        },
    }


def test_kl_for_prefers_matching_ctx():
    row = _row("h1", "Q4_K_M", "f16", 3000, 200, 0.02, 0.08)
    name, kl = pareto._kl_for(row, prefer_ctx=16384)
    assert name == "kld_c16384_n2"
    assert kl == 0.08
    name, kl = pareto._kl_for(row, prefer_ctx=512)
    assert name == "kld_c512_n20"
    assert kl == 0.02


def test_kl_for_falls_back_when_no_match():
    row = _row("h1", "Q4_K_M", "f16", 3000, 200, 0.02, 0.08)
    result = pareto._kl_for(row, prefer_ctx=99999)
    assert result is not None  # falls back to whatever's available


def test_front_identifies_nondominated_speed_quality_tradeoff():
    rows = [
        _row("fast_bad", "Q3_K_M", "q4_0", 1500, 300, 0.5, 0.5),   # fastest, worst quality
        _row("slow_good", "Q8_0", "f16", 4500, 150, 0.01, 0.01),   # slowest, best quality
        _row("dominated", "Q4_K_M", "f16", 3000, 140, 0.4, 0.4),   # slower AND worse than fast_bad
    ]
    for r in rows:
        r["_kl"] = pareto._kl_for(r, prefer_ctx=16384)[1]
    fr = pareto.front(rows, "tg_tps", "_kl", x_max=True, y_min=True)
    names = {r["spec_hash"] for r in fr}
    assert "fast_bad" in names
    assert "slow_good" in names
    assert "dominated" not in names


def test_agreement_strong_correlation_and_recall():
    rows = [
        _row("h1", "Q3_K_M", "q4_0", 1500, 300, 0.50, 0.55),
        _row("h2", "Q4_K_M", "q4_0", 2000, 220, 0.20, 0.25),
        _row("h3", "Q4_K_M", "f16", 3000, 210, 0.05, 0.08),
        _row("h4", "Q6_K", "f16", 3500, 190, 0.02, 0.03),
        _row("h5", "Q8_0", "f16", 4500, 160, 0.01, 0.01),
    ]
    result = pareto.agreement(rows, short_ctx=512, long_ctx=16384, fronts=2, min_n=2, max_n=5)
    assert result["spearman"]["rho"] > 0.9
    assert result["n_configs"] == 5
    assert 0.0 <= result["gate_recall"] <= 1.0


def test_agreement_too_few_configs_returns_error():
    rows = [_row("h1", "Q4_K_M", "f16", 3000, 200, 0.02, 0.08)]
    result = pareto.agreement(rows)
    assert "error" in result
