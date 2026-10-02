import pytest

from harness import splits


def _write(path, text):
    path.write_text(text, encoding="utf-8")
    return path


def test_classify_by_hash_not_path(tmp_path):
    dev = _write(tmp_path / "dev.txt", "alpha paragraph one.\n\nbeta paragraph two.\n")
    manifest = {"version": 1, "text": {}, "tasks": {}}
    splits.record_text_split("dev", dev, manifest_path=tmp_path / "splits.json")
    manifest = splits.load_manifest(tmp_path / "splits.json")

    copy_path = _write(tmp_path / "renamed_copy.txt", dev.read_text())
    assert splits.classify_text(copy_path, manifest) == "dev"
    assert splits.classify_text(dev, manifest) == "dev"

    other = _write(tmp_path / "other.txt", "completely different text.\n")
    assert splits.classify_text(other, manifest) == "adhoc"


def test_load_eval_text_refuses_held_out_without_flag(tmp_path):
    manifest_path = tmp_path / "splits.json"
    held = _write(tmp_path / "held_out.txt", "secret paragraph one.\n\nsecret paragraph two.\n")
    splits.lock_held_out(held, manifest_path=manifest_path)
    manifest = splits.load_manifest(manifest_path)

    with pytest.raises(splits.HeldOutViolation):
        splits.load_eval_text(held, manifest, allow_held_out=False)

    text, split, sha = splits.load_eval_text(held, manifest, allow_held_out=True)
    assert split == "held_out"
    assert len(sha) == 16


def test_load_eval_text_detects_held_out_drift(tmp_path):
    manifest_path = tmp_path / "splits.json"
    held = _write(tmp_path / "held_out.txt", "original content.\n\nmore content.\n")
    splits.lock_held_out(held, manifest_path=manifest_path)

    held.write_text("original content.\n\nTAMPERED.\n", encoding="utf-8")
    manifest = splits.load_manifest(manifest_path)
    with pytest.raises(splits.HeldOutViolation):
        splits.load_eval_text(held, manifest, allow_held_out=True)  # even with the flag -- it drifted


def test_load_eval_text_detects_paragraph_leak(tmp_path):
    manifest_path = tmp_path / "splits.json"
    held = _write(tmp_path / "held_out.txt", "unique held out sentence right here.\n\nanother one.\n")
    splits.lock_held_out(held, manifest_path=manifest_path)

    leaked = _write(tmp_path / "dev.txt",
                     "some dev text.\n\nunique held out sentence right here.\n\nmore dev text.\n")
    manifest = splits.load_manifest(manifest_path)
    with pytest.raises(splits.HeldOutViolation):
        splits.load_eval_text(leaked, manifest, allow_held_out=False)

    # with the flag, a leak is allowed through (final-report override)
    text, split, sha = splits.load_eval_text(leaked, manifest, allow_held_out=True)
    assert split == "adhoc"


def test_load_eval_text_detects_shingle_leak_with_edited_word(tmp_path):
    # Week 4 red-team case: paragraph-hash matching alone misses a leak where
    # one word has been changed -- the 13-gram shingle check catches the
    # unmodified run of words around the edit.
    manifest_path = tmp_path / "splits.json"
    sentence = ("the quick brown fox jumps over the lazy dog while a dozen "
                "curious onlookers watch from the nearby fence line today")
    held = _write(tmp_path / "held_out.txt", f"{sentence}.\n")
    splits.lock_held_out(held, manifest_path=manifest_path)

    edited = sentence.replace("fox", "wolf")  # one word changed, rest identical
    leaked = _write(tmp_path / "dev.txt", f"some other text.\n\n{edited}.\n")
    manifest = splits.load_manifest(manifest_path)

    # paragraph hashes alone would NOT catch this (the paragraph text differs)
    assert set(splits.paragraph_hashes(leaked.read_text())) & set(
        manifest["text"]["held_out"]["paragraph_sha256"]) == set()
    # but the shingle check does
    with pytest.raises(splits.HeldOutViolation):
        splits.load_eval_text(leaked, manifest, allow_held_out=False)


def test_load_eval_text_detects_shingle_leak_merged_paragraphs(tmp_path):
    manifest_path = tmp_path / "splits.json"
    held_text = ("project zephyrion was led by a small team of researchers who worked "
                  "for several years on the underlying problem before publishing\n\n"
                  "their findings were later cited by many other groups across the field")
    held = _write(tmp_path / "held_out.txt", held_text)
    splits.lock_held_out(held, manifest_path=manifest_path)

    # the two held-out paragraphs merged into one, no blank line -- a
    # different paragraph hash than either original, but the same 13-grams
    merged = held_text.replace("\n\n", " ")
    leaked = _write(tmp_path / "search.txt", f"unrelated intro text here.\n\n{merged}\n")
    manifest = splits.load_manifest(manifest_path)
    with pytest.raises(splits.HeldOutViolation):
        splits.load_eval_text(leaked, manifest, allow_held_out=False)


def test_shingles_short_text_returns_empty_set():
    assert splits.shingles("too short") == set()


def test_load_eval_text_clean_dev_passes(tmp_path):
    manifest_path = tmp_path / "splits.json"
    held = _write(tmp_path / "held_out.txt", "held out only sentence.\n")
    splits.lock_held_out(held, manifest_path=manifest_path)
    dev = _write(tmp_path / "dev.txt", "totally unrelated dev content here.\n")
    splits.record_text_split("dev", dev, manifest_path=manifest_path)

    manifest = splits.load_manifest(manifest_path)
    text, split, sha = splits.load_eval_text(dev, manifest, allow_held_out=False)
    assert split == "dev"
    assert text == dev.read_text()


def test_lock_held_out_refuses_silent_relock(tmp_path):
    manifest_path = tmp_path / "splits.json"
    held = _write(tmp_path / "held_out.txt", "version one.\n")
    splits.lock_held_out(held, manifest_path=manifest_path)

    held.write_text("version two.\n", encoding="utf-8")
    with pytest.raises(ValueError):
        splits.lock_held_out(held, manifest_path=manifest_path)

    # relocking with the SAME content is a no-op, not an error
    held.write_text("version one.\n", encoding="utf-8")
    splits.lock_held_out(held, manifest_path=manifest_path)  # should not raise


def test_task_doc_indices_deterministic_and_disjoint(tmp_path):
    manifest_path = tmp_path / "splits.json"
    splits.set_task_split("gsm8k", n_docs=100, fractions={"search": 0.6, "dev": 0.2, "held_out": 0.2},
                           manifest_path=manifest_path)
    manifest = splits.load_manifest(manifest_path)

    search = splits.task_doc_indices("gsm8k", "search", manifest, allow_held_out=False)
    dev = splits.task_doc_indices("gsm8k", "dev", manifest, allow_held_out=False)
    held = splits.task_doc_indices("gsm8k", "held_out", manifest, allow_held_out=True)

    assert len(search) == 60
    assert len(dev) == 20
    assert len(held) == 20
    assert set(search) & set(dev) == set()
    assert set(search) & set(held) == set()
    assert set(dev) & set(held) == set()

    # deterministic: same call again gives the same partition
    assert splits.task_doc_indices("gsm8k", "search", manifest, allow_held_out=False) == search


def test_task_doc_indices_refuses_held_out_without_flag(tmp_path):
    manifest_path = tmp_path / "splits.json"
    splits.set_task_split("gsm8k", n_docs=10, manifest_path=manifest_path)
    manifest = splits.load_manifest(manifest_path)
    with pytest.raises(splits.HeldOutViolation):
        splits.task_doc_indices("gsm8k", "held_out", manifest, allow_held_out=False)


def test_task_doc_indices_unknown_task_raises(tmp_path):
    manifest_path = tmp_path / "splits.json"
    splits.set_task_split("gsm8k", n_docs=10, manifest_path=manifest_path)
    manifest = splits.load_manifest(manifest_path)
    with pytest.raises(KeyError):
        splits.task_doc_indices("some_other_task", "dev", manifest, allow_held_out=False)


def test_ngram_overlap():
    a = "the quick brown fox jumps over the lazy dog again and again today"
    b = "the quick brown fox jumps over the lazy dog again and again today"
    assert splits.ngram_overlap(a, b, n=5) == 1.0
    c = "completely unrelated text with no shared words whatsoever at all here"
    assert splits.ngram_overlap(a, c, n=5) == 0.0


def test_paragraph_hashes_whitespace_normalized():
    a = splits.paragraph_hashes("hello   world this  is a   real paragraph\n\nfoo bar another one here too")
    b = splits.paragraph_hashes("hello world this is a real paragraph\n\nfoo bar another one here too")
    assert a == b
    assert len(a) == 2


def test_paragraphs_drops_short_boilerplate():
    text = "== References ==\n\n|\n\n1/2\n\nThis is an actual real sentence with enough words in it."
    paras = splits.paragraphs(text)
    assert paras == ["This is an actual real sentence with enough words in it."]
