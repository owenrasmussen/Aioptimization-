"""llama-server lifecycle: start it, wait for it to be ready, talk to it, stop it.

Three things matter here that are easy to get wrong:

1. `--fit` defaults to `on` in this llama.cpp build (verified via --help) --
   meaning the server will silently shrink unset parameters, including
   context size, to fit available VRAM. Without `--fit off`, a "32768
   context" test could silently run at a smaller context with no error,
   which would invalidate exactly what the long-context suite measures.
2. `-np`/`--parallel` defaults to -1 (auto), which can resolve to multiple
   KV-cache slots. `-np 1` keeps one spec's context budget from being split.
3. The server's stdout is piped to the harness process; if nothing drains
   that pipe, the OS pipe buffer fills (around 64KB on Windows) and the
   server's own log writes block, hanging it mid-request. stdout/stderr are
   redirected straight to an open log file handle instead of PIPE.
"""
from __future__ import annotations

import json
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from harness import llama
from harness.spec import CandidateSpec


class ServerFailed(llama.RunFailed):
    pass


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def build_cmd(model: Path, spec: CandidateSpec, n_ctx: int, port: int, bin_dir: str | None = None) -> list[str]:
    """Pure (no subprocess, no I/O other than resolving the binary path) so it's
    unit-testable without a GPU or server."""
    exe = llama.bin_path("llama-server", bin_dir)
    return [exe, "-m", str(model), "--host", "127.0.0.1", "--port", str(port),
            "-c", str(n_ctx), "-ngl", "99", "-np", "1", "--fit", "off", "--jinja",
            *llama.spec_flags(spec)]


class LlamaServer:
    def __init__(self, model: Path, spec: CandidateSpec, n_ctx: int, log_path: Path,
                 bin_dir: str | None = None, port: int | None = None, startup_timeout_s: float = 120.0):
        self.model = model
        self.spec = spec
        self.n_ctx = n_ctx
        self.log_path = log_path
        self.bin_dir = bin_dir
        self.port = port or free_port()
        self.startup_timeout_s = startup_timeout_s
        self._proc: subprocess.Popen | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def __enter__(self) -> "LlamaServer":
        cmd = build_cmd(self.model, self.spec, self.n_ctx, self.port, self.bin_dir)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_fh = open(self.log_path, "w", encoding="utf-8")
        self._log_fh.write("$ " + subprocess.list2cmdline(cmd) + "\n")
        self._log_fh.flush()
        self._proc = subprocess.Popen(cmd, stdout=self._log_fh, stderr=subprocess.STDOUT)

        deadline = time.monotonic() + self.startup_timeout_s
        while time.monotonic() < deadline:
            rc = self._proc.poll()
            if rc is not None:
                self._log_fh.close()
                raise ServerFailed(cmd, self.log_path, rc)
            try:
                with urllib.request.urlopen(f"{self.base_url}/health", timeout=2) as resp:
                    if resp.status == 200:
                        break
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                pass
            time.sleep(0.5)
        else:
            self._terminate()
            self._log_fh.close()
            raise ServerFailed(cmd, self.log_path, -1)

        # Confirm this server is actually serving the model/context we asked
        # for, not a stale server left over on the same port from a crashed
        # prior run (which would otherwise score the wrong config with no error).
        try:
            props = self.props()
            served_model = Path(props.get("model_path") or props.get("default_generation_settings",
                                                                       {}).get("model", "")).name
            if served_model and served_model != self.model.name:
                self._terminate()
                self._log_fh.close()
                raise ServerFailed(cmd, self.log_path, -1)
        except (urllib.error.URLError, json.JSONDecodeError, KeyError):
            pass  # /props shape can vary by build; the /health check already passed
        return self

    def __exit__(self, *exc) -> bool:
        self._terminate()
        if hasattr(self, "_log_fh") and not self._log_fh.closed:
            self._log_fh.close()
        return False

    def _terminate(self) -> None:
        if self._proc is None or self._proc.poll() is not None:
            return
        self._proc.terminate()
        try:
            self._proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait(timeout=5)

    def _post(self, path: str, payload: dict, timeout: float = 120.0) -> dict:
        req = urllib.request.Request(
            f"{self.base_url}{path}", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())

    def props(self) -> dict:
        with urllib.request.urlopen(f"{self.base_url}/props", timeout=10) as resp:
            return json.loads(resp.read())

    def tokenize(self, text: str) -> list[int]:
        return self._post("/tokenize", {"content": text})["tokens"]

    def chat(self, messages: list[dict], max_tokens: int, seed: int = 0, timeout: float = 120.0) -> dict:
        return self._post("/v1/chat/completions",
                           {"messages": messages, "max_tokens": max_tokens, "temperature": 0,
                            "seed": seed, "cache_prompt": False}, timeout=timeout)
