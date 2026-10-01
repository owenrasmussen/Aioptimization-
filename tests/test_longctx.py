from harness import longctx


def _fake_tokenize(text: str) -> list[int]:
    return text.split()  # word count stands in for real tokens in these pure tests


PARAS = [f"Paragraph number {i} contains some filler words about nothing in particular here."
         for i in range(50)]
PARA_TOKENS = longctx.tokenize_paragraphs(PARAS, _fake_tokenize)


def test_fill_haystack_never_exceeds_budget():
    budget = 100
    selected = longctx.fill_haystack(PARAS, PARA_TOKENS, budget, offset=0)
    assert sum(longctx.tokenize_paragraphs(selected, _fake_tokenize)) <= budget


def test_fill_haystack_wraps_around_offset():
    selected = longctx.fill_haystack(PARAS, PARA_TOKENS, budget=10_000, offset=45)
    # with a huge budget it should eventually wrap and include early paragraphs too
    assert PARAS[0] in selected
    assert PARAS[45] in selected


def test_insert_at_depth_zero_and_one():
    paras = ["a", "b", "c"]
    assert longctx.insert_at_depth(paras, "X", 0.0)[0] == "X"
    assert longctx.insert_at_depth(paras, "X", 1.0)[-1] == "X"


def test_insert_at_depth_middle():
    paras = ["a", "b", "c", "d"]
    result = longctx.insert_at_depth(paras, "X", 0.5)
    assert result.index("X") == 2


def test_build_needle_contains_expected_code_once():
    trial = longctx.build_needle(PARAS, PARA_TOKENS, target_tokens=500, depth=0.5, seed=0)
    prompt = trial.messages[0]["content"]
    assert trial.expected in prompt
    assert len(longctx.CODE_RE.findall(prompt)) == 1  # exactly one code in a plain needle test


def test_build_needle_deterministic_for_same_seed():
    a = longctx.build_needle(PARAS, PARA_TOKENS, 500, 0.5, seed=7)
    b = longctx.build_needle(PARAS, PARA_TOKENS, 500, 0.5, seed=7)
    assert a.expected == b.expected
    assert a.messages == b.messages


def test_build_needle_different_seeds_differ():
    a = longctx.build_needle(PARAS, PARA_TOKENS, 500, 0.5, seed=1)
    b = longctx.build_needle(PARAS, PARA_TOKENS, 500, 0.5, seed=2)
    assert a.expected != b.expected


def test_build_multihop_requires_the_hop():
    trial = longctx.build_multihop(PARAS, PARA_TOKENS, target_tokens=800, depths=(0.2, 0.7), seed=0,
                                    n_distractors=2)
    prompt = trial.messages[0]["content"]
    # the expected code appears, but so do distractor codes -- a model that just
    # grabs "a code" from the haystack without doing the hop should NOT pass
    codes = longctx.CODE_RE.findall(prompt)
    assert trial.expected in codes
    assert len(codes) > 1  # distractor codes are present


def test_score_exact_match():
    assert longctx.score("The answer is KX-1234567.", "KX-1234567") is True


def test_score_rejects_substring_false_positive():
    # KX-1234567 is a substring of KX-12345678 -- must not count as a match
    assert longctx.score("The code is KX-12345678", "KX-1234567") is False


def test_score_rejects_listing_all_codes():
    response = "The codes mentioned are KX-1111111, KX-2222222, and KX-3333333."
    assert longctx.score(response, "KX-9999999") is False


def test_score_no_code_in_response():
    assert longctx.score("I don't know.", "KX-1234567") is False


def test_make_trials_count():
    trials = longctx.make_trials(PARAS, PARA_TOKENS, lengths=[500, 1000],
                                  needle_depths=(0.1, 0.9), multihop_depth_pairs=((0.2, 0.7),),
                                  seeds=(0, 1))
    needle_trials = [t for t in trials if t.kind == "needle"]
    multihop_trials = [t for t in trials if t.kind == "multihop"]
    # 2 lengths x 2 depths x 2 seeds = 8 needle; 2 lengths x 1 depth-pair x 2 seeds = 4 multihop
    assert len(needle_trials) == 8
    assert len(multihop_trials) == 4
