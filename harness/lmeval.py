"""Task-suite quality evals against a running llama-server.

GSM8K goes through lm-eval-harness's library API (`lm_eval.simple_evaluate`),
imported lazily so pytest and `run.py --no-lm-eval` never pay for it.

HumanEval does NOT go through lm-eval's built-in `humaneval*` tasks at all --
verified live on this machine that loading any of them raises
`NotImplementedError: This metric is currently not supported on Windows`
inside HuggingFace `evaluate`'s `code_eval` metric, and critically that this
fires at task *import* time (inside PyYAML's constructor, loading
lm_eval/tasks/humaneval/utils.py), before `simple_evaluate` even runs --
so `predict_only=True` does not avoid it; there is no way to use that task on
Windows. Instead this module loads the raw `openai/openai_humaneval` dataset
directly, builds the chat prompt and scores pass@1 itself: write
prompt+completion+test+check(entry_point) to a .py file, run it with
`subprocess.run(..., timeout=...)`, which (unlike `evaluate`'s
signal.alarm-based timeout) works the same on Windows and POSIX.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def _primary_metric(results_for_task: dict) -> tuple[str | None, float | None, float | None]:
    """lm-eval's per-task result dict uses compound keys like
    'exact_match,flexible-extract' (metric,filter) plus a matching
    '..._stderr,<filter>' key. Picks the first non-metadata metric key,
    preferring one whose filter contains 'flexible' (the conventional way
    GSM8K is reported) when more than one is present."""
    metric_keys = [k for k in results_for_task if k not in ("alias", "samples", "sample_len", "name")
                   and not k.endswith("_stderr")]
    if not metric_keys:
        return None, None, None
    key = next((k for k in metric_keys if "flexible" in k), metric_keys[0])
    stderr_key = key.replace(",", "_stderr,") if "," in key else f"{key}_stderr"
    return key, results_for_task.get(key), results_for_task.get(stderr_key)


def evaluate_gsm8k(base_url: str, model_alias: str, doc_indices: list[int] | None, out_dir: Path,
                    num_fewshot: int | None = None) -> dict:
    """Runs GSM8K against the server at base_url. doc_indices=None runs the
    full test set; otherwise only those doc indices (from
    harness.splits.task_doc_indices) are evaluated."""
    import lm_eval  # lazy: keeps this an optional dependency for callers that pass --no-lm-eval

    kwargs = dict(
        model="local-chat-completions",
        model_args=(f"model={model_alias},base_url={base_url}/v1/chat/completions,"
                     f"num_concurrent=1,max_retries=3,tokenized_requests=False"),
        tasks=["gsm8k"], apply_chat_template=True, fewshot_as_multiturn=True,
        num_fewshot=num_fewshot, verbosity="ERROR",
    )
    if doc_indices is not None:
        kwargs["samples"] = {"gsm8k": doc_indices}
    results = lm_eval.simple_evaluate(**kwargs)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "gsm8k_raw.json").write_text(json.dumps(results, indent=2, default=str))

    task_result = results["results"]["gsm8k"]
    metric_key, score, stderr = _primary_metric(task_result)
    n_items = len(doc_indices) if doc_indices is not None else task_result.get("sample_len")
    return {"eval_name": "gsm8k", "score": score, "n_items": n_items, "stderr": stderr,
            "details": {"metric_key": metric_key, "raw": task_result,
                        "lm_eval_version": getattr(lm_eval, "__version__", None)}}


_HUMANEVAL_DATASET = None


def _humaneval_dataset():
    global _HUMANEVAL_DATASET
    if _HUMANEVAL_DATASET is None:
        from datasets import load_dataset  # lazy, same reason as lm_eval above
        _HUMANEVAL_DATASET = load_dataset("openai/openai_humaneval")["test"]
    return _HUMANEVAL_DATASET


def _extract_code(response: str, prompt: str) -> str:
    """Pulls the code out of a chat response: prefer a fenced ```python block,
    else assume the whole response is code. Qwen2.5-Instruct reliably fences
    code blocks; this is a reasonable default for other instruct models too."""
    m = _CODE_FENCE_RE.search(response)
    body = m.group(1) if m else response
    # The model's completion is meant to continue `prompt` (which ends mid
    # function signature); if it already repeats the signature, use its
    # version as-is rather than concatenating a duplicate definition.
    if body.strip().startswith("def ") or prompt.strip().split("\n")[-1].strip() in body:
        return body
    return prompt + body


def _run_one_humaneval(server, item: dict, max_tokens: int, timeout_s: float, work_dir: Path) -> dict:
    prompt_text = (f"Complete the following Python function. Respond with only the complete "
                   f"function in a single ```python code block, no explanation.\n\n{item['prompt']}")
    resp = server.chat([{"role": "user", "content": prompt_text}], max_tokens=max_tokens)
    content = resp["choices"][0]["message"]["content"]
    code = _extract_code(content, item["prompt"])

    program = f"{code}\n\n{item['test']}\n\ncheck({item['entry_point']})\n"
    work_dir.mkdir(parents=True, exist_ok=True)
    safe_id = item["task_id"].replace("/", "_")
    # Must be absolute: subprocess.run's cwd=work_dir changes the CHILD's
    # working directory, so a relative program_path would be resolved against
    # work_dir a second time (work_dir/work_dir/<file>.py) and fail to open.
    program_path = (work_dir / f"{safe_id}.py").resolve()
    program_path.write_text(program, encoding="utf-8")

    try:
        proc = subprocess.run([sys.executable, str(program_path)], timeout=timeout_s,
                               capture_output=True, text=True, cwd=work_dir)
        passed, error = proc.returncode == 0, (proc.stderr[-2000:] if proc.returncode != 0 else None)
    except subprocess.TimeoutExpired:
        passed, error = False, f"timed out after {timeout_s}s"

    return {"task_id": item["task_id"], "passed": passed, "error": error, "response": content,
            "program_path": str(program_path)}


def evaluate_humaneval(server, doc_indices: list[int] | None, out_dir: Path,
                        max_tokens: int = 512, timeout_s: float = 10.0) -> dict:
    """Runs HumanEval (pass@1) against `server` directly, bypassing lm-eval's
    built-in task entirely -- see module docstring. doc_indices=None runs all
    164 problems."""
    ds = _humaneval_dataset()
    indices = doc_indices if doc_indices is not None else range(len(ds))
    work_dir = out_dir / "humaneval_programs"
    results = [_run_one_humaneval(server, ds[i], max_tokens, timeout_s, work_dir) for i in indices]

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "humaneval_samples.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps({k: v for k, v in r.items() if k != "response"} | {"response": r["response"]}) + "\n")

    n = len(results)
    n_pass = sum(1 for r in results if r["passed"])
    score = n_pass / n if n else None
    stderr = (score * (1 - score) / n) ** 0.5 if n and score is not None else None
    return {"eval_name": "humaneval", "score": score, "n_items": n, "stderr": stderr,
            "details": {"n_pass": n_pass, "samples_path": str(out_dir / "humaneval_samples.jsonl")}}
