"""
Core data models for EvalKit.

Every backend (ollama, llama.cpp, OpenAI, etc.) produces an InferenceResult.
That single shape is what the rest of the system — analysis, recommendations,
reporters — consumes. Backends are responsible for translating their raw output
into this structure.
"""

from dataclasses import dataclass, field
from typing import Optional
import time


@dataclass
class InferenceResult:
    """
    The output of a single inference run, normalized across all backends.

    All durations are in seconds (float). We convert from nanoseconds,
    milliseconds, or whatever the backend uses — internally we always use seconds.
    """

    # --- Identity ---
    backend: str              # e.g. "ollama", "llamacpp", "openai"
    model: str                # e.g. "llama3.2:3b", "gpt-4o-mini"
    prompt: str               # the exact prompt sent

    # --- The response ---
    response: str             # the generated text
    prompt_tokens: int        # how many tokens the prompt consumed
    completion_tokens: int    # how many tokens the model generated

    # --- Core timing (seconds) ---
    # load_duration: time to load model from disk into memory.
    # Only relevant on cold starts. Warm runs will have this near 0.
    load_duration: float = 0.0

    # prompt_eval_duration: time to process/encode the prompt tokens.
    # This is the "prefill" phase — the model reads your entire prompt
    # before generating a single output token.
    prompt_eval_duration: float = 0.0

    # eval_duration: time spent generating the completion tokens.
    # This is the "decode" phase — one token generated per forward pass.
    eval_duration: float = 0.0

    # total_duration: wall-clock time for the entire request.
    # Should roughly equal load + prompt_eval + eval, but may differ
    # due to network overhead, queuing, etc.
    total_duration: float = 0.0

    # --- Derived metrics (computed automatically) ---
    # We use __post_init__ to compute these so callers don't have to.

    # tokens per second during the generation (decode) phase
    tokens_per_second: float = field(init=False)

    # tokens per second during the prompt processing (prefill) phase
    prompt_tokens_per_second: float = field(init=False)

    # time to first token — the latency a user "feels" before anything appears.
    # For non-streaming: load + prompt_eval (no output until full generation done)
    # For streaming: load + prompt_eval (first chunk arrives after prefill)
    time_to_first_token: float = field(init=False)

    # --- Optional metadata ---
    timestamp: float = field(default_factory=time.time)
    iteration: int = 1        # which iteration this was in a multi-run benchmark
    error: Optional[str] = None

    def __post_init__(self):
        # Compute tokens per second for generation phase.
        # Guard against zero duration to avoid division by zero.
        if self.eval_duration > 0 and self.completion_tokens > 0:
            self.tokens_per_second = self.completion_tokens / self.eval_duration
        else:
            self.tokens_per_second = 0.0

        # Compute tokens per second for prompt processing (prefill) phase.
        if self.prompt_eval_duration > 0 and self.prompt_tokens > 0:
            self.prompt_tokens_per_second = self.prompt_tokens / self.prompt_eval_duration
        else:
            self.prompt_tokens_per_second = 0.0

        # TTFT = load time + prefill time.
        # On a warm run (model already loaded), load_duration ≈ 0,
        # so TTFT ≈ prompt_eval_duration. That's the real latency floor.
        self.time_to_first_token = self.load_duration + self.prompt_eval_duration


@dataclass
class BenchmarkResult:
    """
    Aggregated result from running the same prompt N times.

    We run multiple iterations to get stable measurements — a single
    run can be noisy due to OS scheduling, memory pressure, etc.
    """
    results: list[InferenceResult]

    # Derived stats across all iterations (computed in __post_init__)
    mean_tokens_per_second: float = field(init=False)
    mean_ttft: float = field(init=False)
    mean_total_duration: float = field(init=False)

    # Percentile latencies — p95 tells you the "worst case a user will hit
    # 95% of the time". More useful than mean for user-facing systems.
    p50_total_duration: float = field(init=False)
    p95_total_duration: float = field(init=False)
    p99_total_duration: float = field(init=False)

    def __post_init__(self):
        import statistics

        tps = [r.tokens_per_second for r in self.results if r.tokens_per_second > 0]
        ttfts = [r.time_to_first_token for r in self.results]
        totals = sorted([r.total_duration for r in self.results])

        self.mean_tokens_per_second = statistics.mean(tps) if tps else 0.0
        self.mean_ttft = statistics.mean(ttfts) if ttfts else 0.0
        self.mean_total_duration = statistics.mean(totals) if totals else 0.0

        n = len(totals)
        self.p50_total_duration = totals[int(n * 0.50)] if n else 0.0
        self.p95_total_duration = totals[int(n * 0.95)] if n else 0.0
        self.p99_total_duration = totals[int(n * 0.99)] if n else 0.0
