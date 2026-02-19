"""
Recommendations engine for EvalKit.

This is the core differentiator. It takes an InferenceResult + HardwareProfile
and produces a ranked list of Recommendations — specific, actionable advice
for making inference faster.

Architecture:
  - Each rule is a private method that returns List[Recommendation]
  - Empty list means "rule doesn't apply to this result"
  - All rules run, results are merged and sorted by impact
  - Easy to add new rules: write a method, add it to _all_rules()

Impact levels:
  - high   → likely 2x+ improvement
  - medium → likely 20-50% improvement
  - low    → minor improvement or informational
"""

from dataclasses import dataclass
from typing import Literal, Optional
from evalkit.core.metrics import InferenceResult
from evalkit.core.hardware import HardwareProfile

Impact = Literal["high", "medium", "low"]
Category = Literal["hardware", "memory", "context", "quantization", "model"]

# Thresholds that trigger rules
SLOW_TPS_THRESHOLD        = 20.0   # tok/s below this on capable hardware = flag it
VERY_SLOW_TPS_THRESHOLD   = 8.0    # tok/s below this = definitely something wrong
COLD_START_THRESHOLD      = 1.0    # load_duration above this = cold start
SLOW_PREFILL_THRESHOLD    = 2.0    # prompt_eval_duration above this = slow prefill
LONG_PROMPT_THRESHOLD     = 200    # prompt_tokens above this = worth flagging
METAL_EXPECTED_TPS        = 25.0   # minimum expected tok/s on Apple Silicon with Metal


@dataclass
class Recommendation:
    """
    A single optimization recommendation.

    title:              Short label shown in the summary line
    description:        What's happening and why it matters
    action:             Exact step to take — no vague advice
    impact:             high / medium / low
    estimated_speedup:  Human-readable estimate e.g. "5-10x", "~30%"
                        None if we can't honestly estimate
    category:           What part of the system this addresses
    """
    title: str
    description: str
    action: str
    impact: Impact
    category: Category
    estimated_speedup: Optional[str] = None


class RecommendationEngine:
    """
    Runs all rules against an InferenceResult and returns ranked recommendations.

    Usage:
        engine = RecommendationEngine()
        recs = engine.analyze(result, hardware)
    """

    def analyze(
        self,
        result: InferenceResult,
        hardware: HardwareProfile,
    ) -> list[Recommendation]:
        """
        Run all rules and return recommendations sorted by impact.
        High impact first, then medium, then low.
        """
        all_recs = []
        for rule in self._all_rules():
            all_recs.extend(rule(result, hardware))

        # Sort: high → medium → low
        order = {"high": 0, "medium": 1, "low": 2}
        return sorted(all_recs, key=lambda r: order[r.impact])

    def _all_rules(self):
        """Return all rule methods. Add new rules here."""
        return [
            self._rule_metal_not_used,
            self._rule_cold_start,
            self._rule_slow_prefill,
            self._rule_slow_generation,
            self._rule_long_prompt,
            self._rule_cpu_only_no_gpu,
        ]

    # ── Rules ────────────────────────────────────────────────────────────────

    def _rule_metal_not_used(
        self,
        result: InferenceResult,
        hardware: HardwareProfile,
    ) -> list[Recommendation]:
        """
        Apple Silicon detected but generation is slower than Metal should be.

        On Apple Silicon, ollama uses Metal (GPU) by default. But if tok/s
        is well below what Metal delivers, the GPU may not be engaged —
        possibly due to an old ollama version or a model format issue.

        Metal on M-series: even M1 does 25-40 tok/s on a 3B model.
        CPU-only on same chip: 5-10 tok/s.
        """
        if not hardware.is_apple_silicon:
            return []

        if result.tokens_per_second >= METAL_EXPECTED_TPS:
            return []   # already fast, Metal is working fine

        return [Recommendation(
            title="Metal GPU may not be active",
            description=(
                f"You're on Apple Silicon ({hardware.cpu_name or 'Apple chip'}) "
                f"but generation is only {result.tokens_per_second:.1f} tok/s. "
                f"Metal-accelerated inference on this chip should be "
                f"{METAL_EXPECTED_TPS:.0f}+ tok/s on a 3B model. "
                f"The GPU may not be engaged."
            ),
            action=(
                "1. Update ollama to the latest version: brew upgrade ollama\n"
                "2. Restart ollama: killall ollama && ollama serve\n"
                "3. Re-run your benchmark — Metal should activate automatically.\n"
                "4. Confirm GPU is active: ollama ps (look for '100% GPU' next to model)"
            ),
            impact="high",
            category="hardware",
            estimated_speedup="3-5x",
        )]

    def _rule_cold_start(
        self,
        result: InferenceResult,
        hardware: HardwareProfile,
    ) -> list[Recommendation]:
        """
        Model had to load from disk — first request always pays this cost.

        Not a bug, but worth flagging because it inflates TTFT dramatically.
        The fix is keeping the model warm between requests.
        """
        if result.load_duration <= COLD_START_THRESHOLD:
            return []

        return [Recommendation(
            title="Cold start — model loaded from disk",
            description=(
                f"This run spent {result.load_duration:.1f}s loading the model into memory "
                f"before any inference began. This only happens on the first request after "
                f"ollama unloads the model. Subsequent requests skip this cost entirely."
            ),
            action=(
                "To keep the model warm between requests, set keep_alive in your payload:\n"
                '  {"model": "llama3.2:3b", "keep_alive": -1, ...}\n'
                "keep_alive: -1 means never unload. keep_alive: '10m' means 10 minutes.\n"
                "Default is 5 minutes of inactivity before ollama unloads the model."
            ),
            impact="medium",
            category="memory",
            estimated_speedup=f"saves {result.load_duration:.1f}s on first request",
        )]

    def _rule_slow_prefill(
        self,
        result: InferenceResult,
        hardware: HardwareProfile,
    ) -> list[Recommendation]:
        """
        Prefill (prompt processing) is slow.

        Prefill is typically fast because it runs in parallel across all tokens.
        If it's slow, it usually means:
          - The prompt is very long (more tokens = more compute)
          - Hardware isn't being used efficiently (CPU instead of GPU)
          - The model is being re-evaluated when it could be cached
        """
        if result.prompt_eval_duration <= SLOW_PREFILL_THRESHOLD:
            return []

        # Only fire this if it's NOT already caught by the Metal rule
        # (slow prefill on Apple Silicon without Metal is the same root cause)
        if hardware.is_apple_silicon and result.tokens_per_second < METAL_EXPECTED_TPS:
            return []   # Metal rule covers this more specifically

        recs = []

        if result.prompt_tokens > LONG_PROMPT_THRESHOLD:
            recs.append(Recommendation(
                title="Long prompt is slowing prefill",
                description=(
                    f"Your prompt is {result.prompt_tokens} tokens and took "
                    f"{result.prompt_eval_duration:.2f}s to process. "
                    f"Prefill time scales linearly with prompt length — "
                    f"every extra token adds compute before the first output token appears."
                ),
                action=(
                    "1. Trim your system prompt — remove boilerplate, redundant instructions.\n"
                    "2. If reusing the same prompt prefix, look into prefix caching (ollama supports this).\n"
                    "3. Consider whether all context in the prompt is actually needed."
                ),
                impact="medium",
                category="context",
                estimated_speedup="proportional to prompt reduction",
            ))
        else:
            recs.append(Recommendation(
                title="Slow prefill for prompt size",
                description=(
                    f"Prefill took {result.prompt_eval_duration:.2f}s for only "
                    f"{result.prompt_tokens} tokens "
                    f"({result.prompt_tokens_per_second:.1f} tok/s). "
                    f"This is slower than expected and likely a hardware utilization issue."
                ),
                action=(
                    "Check that your inference backend is using available GPU acceleration.\n"
                    "For ollama on Linux with NVIDIA: ensure CUDA drivers are installed.\n"
                    "Run: ollama ps — look for GPU % next to your model."
                ),
                impact="medium",
                category="hardware",
                estimated_speedup="2-4x with GPU acceleration",
            ))

        return recs

    def _rule_slow_generation(
        self,
        result: InferenceResult,
        hardware: HardwareProfile,
    ) -> list[Recommendation]:
        """
        Generation (decode) speed is very slow even accounting for hardware.

        We only fire this on non-Apple-Silicon machines (or Apple Silicon
        where Metal appears to be working) to avoid duplicate advice.
        """
        if result.tokens_per_second <= 0:
            return []

        # On Apple Silicon, the Metal rule handles slow generation
        if hardware.is_apple_silicon:
            return []

        if result.tokens_per_second >= SLOW_TPS_THRESHOLD:
            return []   # fast enough, no recommendation needed

        if result.tokens_per_second < VERY_SLOW_TPS_THRESHOLD:
            return [Recommendation(
                title="Generation is very slow — check quantization",
                description=(
                    f"Generation speed is {result.tokens_per_second:.1f} tok/s. "
                    f"On CPU-only inference, lighter quantization reduces the "
                    f"memory bandwidth needed per token — which directly improves speed."
                ),
                action=(
                    "Try a more aggressively quantized version of the model:\n"
                    "  Q8_0  → good quality, smaller than fp16\n"
                    "  Q4_K_M → best quality-per-size tradeoff (recommended)\n"
                    "  Q3_K_M → faster, small quality loss\n"
                    "In ollama: pull a quantized variant or specify in Modelfile."
                ),
                impact="medium",
                category="quantization",
                estimated_speedup="20-40% depending on quant level",
            )]

        return []

    def _rule_long_prompt(
        self,
        result: InferenceResult,
        hardware: HardwareProfile,
    ) -> list[Recommendation]:
        """
        Prompt is long. Flag it with context on why that matters for TTFT.
        Only fires if prefill isn't already slow (that rule covers it more specifically).
        """
        if result.prompt_tokens <= LONG_PROMPT_THRESHOLD:
            return []

        if result.prompt_eval_duration > SLOW_PREFILL_THRESHOLD:
            return []   # slow_prefill rule already covers this

        return [Recommendation(
            title="Long prompt — watch TTFT at scale",
            description=(
                f"Your prompt is {result.prompt_tokens} tokens. "
                f"Prefill is fast now ({result.prompt_eval_duration:.2f}s), "
                f"but TTFT scales linearly with prompt length. "
                f"If you add a large system prompt or long context, this will grow."
            ),
            action=(
                "No action needed now. Keep an eye on this if you extend the prompt.\n"
                "As a benchmark: 1000-token prompt adds ~1s prefill on CPU, ~0.1s on GPU."
            ),
            impact="low",
            category="context",
            estimated_speedup=None,
        )]

    def _rule_cpu_only_no_gpu(
        self,
        result: InferenceResult,
        hardware: HardwareProfile,
    ) -> list[Recommendation]:
        """
        Machine has no GPU (no Metal, no CUDA) — pure CPU inference.
        Give honest expectations and the best CPU-specific advice.
        """
        if hardware.has_metal or hardware.has_cuda:
            return []   # GPU is available, other rules handle optimization

        if result.tokens_per_second <= 0:
            return []

        return [Recommendation(
            title="CPU-only inference — no GPU detected",
            description=(
                f"No GPU acceleration (Metal or CUDA) was detected. "
                f"Inference is running entirely on CPU at {result.tokens_per_second:.1f} tok/s. "
                f"CPU inference is significantly slower than GPU for LLMs — "
                f"this is expected, not a configuration error."
            ),
            action=(
                "To improve CPU inference speed:\n"
                "1. Use a smaller or more quantized model (Q4_K_M is best for CPU)\n"
                "2. Increase thread count in ollama: OLLAMA_NUM_THREADS=8 ollama serve\n"
                "3. Close other applications to reduce memory pressure\n"
                "4. Consider a machine with Apple Silicon or NVIDIA GPU for 5-20x speedup"
            ),
            impact="low",
            category="hardware",
            estimated_speedup="20-30% with tuning; 5-20x with GPU hardware",
        )]
