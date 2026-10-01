from pathlib import Path

from harness.lmeval import _extract_code, _primary_metric, _run_one_humaneval


def test_primary_metric_prefers_flexible_extract():
    # Real shape returned by lm_eval 0.4.13 for gsm8k, captured live against
    # Qwen2.5-3B-Instruct via LlamaServer.
    result = {
        "name": "gsm8k", "alias": "gsm8k", "sample_len": 5,
        "exact_match,strict-match": 0.2, "exact_match_stderr,strict-match": 0.2,
        "exact_match,flexible-extract": 1.0, "exact_match_stderr,flexible-extract": 0.0,
    }
    key, score, stderr = _primary_metric(result)
    assert key == "exact_match,flexible-extract"
    assert score == 1.0
    assert stderr == 0.0


def test_primary_metric_falls_back_without_flexible():
    result = {"name": "x", "alias": "x", "my_metric,default": 0.75, "my_metric_stderr,default": 0.1}
    key, score, stderr = _primary_metric(result)
    assert key == "my_metric,default"
    assert score == 0.75
    assert stderr == 0.1


def test_primary_metric_empty_results():
    assert _primary_metric({"alias": "x", "sample_len": 0}) == (None, None, None)


def test_extract_code_from_fenced_block():
    response = "Here's the solution:\n```python\ndef add(a, b):\n    return a + b\n```\nDone."
    code = _extract_code(response, "def add(a, b):")
    assert "def add(a, b):" in code
    assert "return a + b" in code
    assert "```" not in code


def test_extract_code_no_fence_assumes_whole_response_is_code():
    response = "def add(a, b):\n    return a + b"
    code = _extract_code(response, "def add(a, b):")
    assert code == response


def test_extract_code_prepends_prompt_if_completion_only():
    prompt = "def add(a: int, b: int) -> int:\n    "
    response = "```python\nreturn a + b\n```"
    code = _extract_code(response, prompt)
    assert "def add" in code
    assert "return a + b" in code


class _FakeServer:
    def __init__(self, response_content: str):
        self._content = response_content

    def chat(self, messages, max_tokens):
        return {"choices": [{"message": {"content": self._content}}]}


def test_run_one_humaneval_executes_with_relative_work_dir(tmp_path, monkeypatch):
    # Regression test: subprocess.run's cwd=work_dir plus a RELATIVE program
    # path doubled the directory (work_dir/work_dir/<file>.py) and crashed
    # with "can't open file" -- caught live, not by inspection. tmp_path is
    # always absolute, so the bug only reproduces with a relative work_dir
    # (as in real usage, where run_dir is a relative "results/runs/..."
    # path) -- chdir into tmp_path and pass a relative work_dir to match.
    monkeypatch.chdir(tmp_path)
    item = {"task_id": "HumanEval/0", "entry_point": "add",
            "prompt": "def add(a, b):\n    ", "test": "def check(candidate):\n    assert candidate(2, 3) == 5"}
    server = _FakeServer("```python\ndef add(a, b):\n    return a + b\n```")
    result = _run_one_humaneval(server, item, max_tokens=100, timeout_s=5, work_dir=Path("programs"))
    assert result["passed"] is True
    assert result["error"] is None


def test_run_one_humaneval_reports_failure_for_wrong_code(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    item = {"task_id": "HumanEval/1", "entry_point": "add",
            "prompt": "def add(a, b):\n    ", "test": "def check(candidate):\n    assert candidate(2, 3) == 5"}
    server = _FakeServer("```python\ndef add(a, b):\n    return a - b\n```")  # wrong on purpose
    result = _run_one_humaneval(server, item, max_tokens=100, timeout_s=5, work_dir=Path("programs"))
    assert result["passed"] is False
    assert result["error"]
