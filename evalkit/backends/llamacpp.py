"""
llama.cpp backend adapter.

Unlike the ollama backend (HTTP request → JSON response), this backend
spawns a subprocess — runs the llama-completion binary directly, captures
its stdout and stderr, then parses the output.

Key differences from ollama:
  - We manage the process lifecycle (spawn, wait, kill on timeout)
  - Timing data lives on stderr, model response lives on stdout
  - Output format varies between llama.cpp versions — we handle both
  - No server to keep warm — every run is a fresh process (always cold start)
  - Context size must be set explicitly or llama.cpp defaults to 131072,
    which allocates ~14GB of KV cache and tanks performance on most machines
  - llama-completion is a one-shot binary (no --no-cnv needed, unlike llama-cli)

Supported output formats:
  - New (build b8000+):  "common_perf_print:  load time = X ms"
  - Old (pre-b8000):     "llama_print_timings: load time = X ms"
"""

import re
import shutil
import subprocess
import time
import os
from typing import Optional

from evalkit.core.metrics import InferenceResult

MILLISECONDS = 1_000  # divide by this to convert ms → seconds

# Timing line patterns — match both old and new llama.cpp output formats.
# Each pattern captures the numeric value(s) we need.
_TIMING_PREFIX = r"(?:common_perf_print|llama_print_timings)\s*:"

PATTERN_LOAD = re.compile(
    _TIMING_PREFIX + r"\s+load time\s*=\s*([\d.]+)\s*ms"
)
PATTERN_PROMPT_EVAL = re.compile(
    _TIMING_PREFIX + r"\s+prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens"
)
PATTERN_EVAL = re.compile(
    _TIMING_PREFIX + r"\s+eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*runs"
)
PATTERN_TOTAL = re.compile(
    _TIMING_PREFIX + r"\s+total time\s*=\s*([\d.]+)\s*ms"
)

# Binary search order — try these locations in sequence
BINARY_CANDIDATES = [
    "llama-completion",          # brew install llama.cpp (new)
    "/usr/local/bin/llama-completion",
    "/opt/homebrew/bin/llama-completion",
    "llama-cli",                 # some older installs
    "/usr/local/bin/llama-cli",
    "/opt/homebrew/bin/llama-cli",
]


class LlamaCppBackend:
    """
    Wraps the llama-completion (or llama-cli) binary as a subprocess.

    Parameters:
        binary:       Path to llama-completion binary. Auto-detected if None.
        context_size: KV cache context window in tokens. Default 4096.
                      WARNING: llama.cpp defaults to 131072 which allocates
                      ~14GB of KV cache. Always set this explicitly.
        threads:      CPU threads. Defaults to half the logical core count.
        n_gpu_layers: GPU layers to offload (-ngl flag). 0 = CPU only.
                      None = let llama.cpp decide.
        timeout:      Max seconds to wait for a subprocess to complete.
    """

    def __init__(
        self,
        binary: Optional[str] = None,
        context_size: int = 4096,
        threads: Optional[int] = None,
        n_gpu_layers: Optional[int] = None,
        timeout: float = 300.0,
    ):
        self.binary = binary or self._find_binary()
        self.context_size = context_size
        self.threads = threads or max(1, (os.cpu_count() or 4) // 2)
        self.n_gpu_layers = n_gpu_layers
        self.timeout = timeout

    def is_available(self) -> bool:
        """Check if the binary exists and is executable."""
        try:
            result = subprocess.run(
                [self.binary, "--help"],
                capture_output=True,
                timeout=5,
            )
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def run(self, model_path: str, prompt: str, iteration: int = 1) -> InferenceResult:
        """
        Run a single inference and return a normalized InferenceResult.

        model_path: full path to a .gguf model file
        prompt:     the text prompt to send
        iteration:  which run this is in a benchmark sequence
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                f"Make sure you're pointing at a .gguf file."
            )

        cmd = self._build_command(model_path, prompt)
        wall_start = time.perf_counter()

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"llama-completion timed out after {self.timeout}s. "
                f"Try a smaller model or increase --timeout."
            )

        wall_end = time.perf_counter()

        response = _extract_response(proc.stdout, prompt)
        timing = _parse_timing(proc.stderr)

        return InferenceResult(
            backend="llamacpp",
            model=os.path.basename(model_path),  # just filename, not full path
            prompt=prompt,
            response=response,
            prompt_tokens=timing["prompt_tokens"],
            completion_tokens=timing["eval_tokens"],
            load_duration=timing["load_ms"] / MILLISECONDS,
            prompt_eval_duration=timing["prompt_eval_ms"] / MILLISECONDS,
            eval_duration=timing["eval_ms"] / MILLISECONDS,
            total_duration=timing["total_ms"] / MILLISECONDS,
            iteration=iteration,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_command(self, model_path: str, prompt: str) -> list[str]:
        """Construct the subprocess command with all flags."""
        cmd = [
            self.binary,
            "--model",        model_path,
            "--prompt",       prompt,
            "--ctx-size",     str(self.context_size),  # cap KV cache allocation
            "--threads",      str(self.threads),
            "--n-predict",    "512",                   # max tokens to generate
            # NOTE: do not use --log-disable — it also silences timing lines on stderr
        ]

        if self.n_gpu_layers is not None:
            cmd.extend(["--n-gpu-layers", str(self.n_gpu_layers)])

        return cmd

    def _find_binary(self) -> str:
        """Search for the llama-completion binary in common locations."""
        for candidate in BINARY_CANDIDATES:
            if shutil.which(candidate) or os.path.isfile(candidate):
                return candidate
        raise RuntimeError(
            "llama-completion binary not found.\n"
            "Install with: brew install llama.cpp\n"
            "Or specify the path with: LlamaCppBackend(binary='/path/to/llama-completion')"
        )


# ── Parsing helpers (module-level, easier to test) ────────────────────────────

def _extract_response(stdout: str, prompt: str) -> str:
    """
    Extract just the model's generated text from llama-completion stdout.

    llama-completion wraps output in a conversation template when the model
    has a chat template (most instruct models do):

        user

        {prompt}assistant

        {response}

        > EOF by user

    We extract everything between "assistant\n\n" and the first "\n>" or end.
    Falls back to stripping the prompt prefix if no template markers found.
    """
    # Try conversation template format (instruct models)
    if "assistant\n\n" in stdout:
        parts = stdout.split("assistant\n\n", 1)
        if len(parts) == 2:
            response = parts[1]
            # Strip the "> EOF by user" trailer and anything after it
            if "\n>" in response:
                response = response[:response.index("\n>")]
            return response.strip()

    # Fallback: strip the prompt from the beginning of stdout
    # (base models without chat templates just continue the text)
    if stdout.startswith(prompt):
        return stdout[len(prompt):].strip()

    return stdout.strip()


def _parse_timing(stderr: str) -> dict:
    """
    Parse timing values from llama.cpp stderr output.

    Returns a dict with all timing values in milliseconds.
    Defaults to 0 for any value that isn't found — we degrade
    gracefully rather than crash on unexpected output formats.
    """
    result = {
        "load_ms": 0.0,
        "prompt_eval_ms": 0.0,
        "prompt_tokens": 0,
        "eval_ms": 0.0,
        "eval_tokens": 0,
        "total_ms": 0.0,
    }

    m = PATTERN_LOAD.search(stderr)
    if m:
        result["load_ms"] = float(m.group(1))

    m = PATTERN_PROMPT_EVAL.search(stderr)
    if m:
        result["prompt_eval_ms"] = float(m.group(1))
        result["prompt_tokens"] = int(m.group(2))

    m = PATTERN_EVAL.search(stderr)
    if m:
        result["eval_ms"] = float(m.group(1))
        result["eval_tokens"] = int(m.group(2))

    m = PATTERN_TOTAL.search(stderr)
    if m:
        result["total_ms"] = float(m.group(1))

    return result
