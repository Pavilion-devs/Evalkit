"""
Ollama backend adapter.

Ollama exposes a local REST API at http://localhost:11434.
This module calls that API, captures the raw response, and translates
it into an InferenceResult that the rest of EvalKit understands.

Ollama's timing values come back in nanoseconds. We convert to seconds.
"""

import httpx
import time
from evalkit.core.metrics import InferenceResult

OLLAMA_BASE_URL = "http://localhost:11434"
NANOSECONDS = 1_000_000_000  # we divide by this to convert ns → seconds


class OllamaBackend:
    """
    Wraps ollama's /api/generate endpoint.

    Why httpx and not urllib? httpx gives us a cleaner API and is
    async-ready for when we add concurrent benchmarking later.
    For now we use it synchronously.
    """

    def __init__(self, base_url: str = OLLAMA_BASE_URL, timeout: float = 120.0):
        self.base_url = base_url
        self.timeout = timeout

    def is_available(self) -> bool:
        """Check if the ollama server is running."""
        try:
            resp = httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
            return resp.status_code == 200
        except httpx.ConnectError:
            return False

    def list_models(self) -> list[str]:
        """Return names of all locally available models."""
        resp = httpx.get(f"{self.base_url}/api/tags", timeout=10.0)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]

    def run(self, model: str, prompt: str, iteration: int = 1) -> InferenceResult:
        """
        Send a single inference request and return a normalized InferenceResult.

        We use stream=False to get all timing data in one response object.
        Ollama only provides detailed timing breakdowns in the non-streaming
        response — streaming gives us TTFT but loses the prefill timing detail.
        We'll add streaming support later for TTFT-focused benchmarks.
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }

        wall_start = time.perf_counter()

        try:
            resp = httpx.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except httpx.ConnectError:
            raise RuntimeError(
                "Cannot connect to ollama. Is the ollama app running?"
            )
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"Ollama returned an error: {e.response.text}")

        wall_end = time.perf_counter()
        raw = resp.json()

        # Ollama returns durations in nanoseconds — convert everything to seconds.
        # If a field is missing (older ollama versions), default to 0.
        return InferenceResult(
            backend="ollama",
            model=model,
            prompt=prompt,
            response=raw.get("response", ""),
            prompt_tokens=raw.get("prompt_eval_count", 0),
            completion_tokens=raw.get("eval_count", 0),
            load_duration=raw.get("load_duration", 0) / NANOSECONDS,
            prompt_eval_duration=raw.get("prompt_eval_duration", 0) / NANOSECONDS,
            eval_duration=raw.get("eval_duration", 0) / NANOSECONDS,
            total_duration=raw.get("total_duration", 0) / NANOSECONDS,
            iteration=iteration,
        )
