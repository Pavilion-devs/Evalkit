# EvalKit

**Inference profiler and optimizer for local and API-based LLMs.**

Most LLM tools tell you what your model said. EvalKit tells you how fast it said it — and exactly what to change to make it faster.

```
Running inference on llama3.2:3b...
Hardware: Intel Core i7 · CPU only · 16GB RAM

╭─────────────────── EvalKit Profile ───────────────────╮
│ Model:    llama3.2:3b  (ollama)                       │
│ Prompt:   "What is the KV cache in a transformer?"    │
│ Response: "The KV cache stores key-value pairs..."    │
╰───────────────────────────────────────────────────────╯

  Timing
  ──────────────────────────────────────────────────────
  Load          4.508s   ████░░░░░░░░░░   COLD START
  Prefill       1.992s   ██░░░░░░░░░░░░   35 tokens in
  Generation   17.909s   ██████████████   233 tokens out

  TTFT    6.500s  SLOW  (includes cold load)
  TTFT (warm est.)  1.992s  OK

  Throughput
  ──────────────────────────────────────────────────────
  Generation   13.0 tok/s   █████░░░░░░░░░   OK
  Prefill      17.6 tok/s   ███████░░░░░░░   OK

  Recommendations
  ──────────────────────────────────────────────────────
  [MED]  Cold start — model loaded from disk
         Estimated gain: saves 4.5s on first request
         → Set keep_alive: -1 in your request payload

  [LOW]  CPU-only inference — no GPU detected
         Estimated gain: 5-20x with GPU hardware
         → OLLAMA_NUM_THREADS=8 ollama serve
```

---

## Why EvalKit

The existing LLM observability landscape (LangSmith, Helicone, Braintrust) is built for API-based workflows. None of them tell you *why* your local model is slow or what to change.

EvalKit is built specifically for developers running their own models — on llama.cpp, ollama, Gemma CPP, or self-hosted serving frameworks. It separates the inference timeline into phases (load → prefill → generation), measures each one, and gives you ranked, hardware-aware recommendations with exact steps to take.

Not a dashboard. Not a logger. A debugger for inference.

---

## Installation

**Requirements:** Python 3.10+, [ollama](https://ollama.com) installed and running.

```bash
git clone https://github.com/yourusername/evalkit.git
cd evalkit
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Pull a model to test with:

```bash
ollama pull llama3.2:3b
```

---

## Usage

### Profile a single run

```bash
evalkit profile --model llama3.2:3b --prompt "Explain what a transformer is."
```

Shows timing breakdown (load / prefill / generation), throughput (tok/s), TTFT, and ranked optimization recommendations based on your hardware.

### Benchmark across multiple runs

```bash
evalkit benchmark --model llama3.2:3b --prompt "Your prompt here." --iterations 10
```

Runs the same prompt N times and shows latency distribution — mean, p50, p95, p99. Use this to understand variance, not just average performance. p95 tells you what a user having a bad day experiences. p99 tells you your worst case.

---

## What the metrics mean

| Metric | What it measures |
|--------|-----------------|
| **Load** | Time to load model from disk into RAM. Cold start only — warm runs skip this. |
| **Prefill** | Time to process your prompt tokens. Scales linearly with prompt length. |
| **Generation** | Time to produce output tokens. Sequential — one token per forward pass, cannot be parallelized. |
| **TTFT** | Time to first token = load + prefill. What the user feels before any output appears. |
| **tok/s** | Tokens generated per second during the decode phase. The primary throughput metric. |
| **p50 / p95 / p99** | Latency percentiles across N runs. p95 = 95% of runs were faster than this value. |

---

## Supported backends

| Backend | Status |
|---------|--------|
| ollama | ✅ Supported |
| llama.cpp | Coming soon |
| Gemma CPP | Coming soon |
| OpenAI API | Coming soon |
| Anthropic API | Coming soon |
| vLLM | Coming soon |

---

## Recommendations engine

After every profile or benchmark run, EvalKit analyzes your results against your hardware and fires rules ranked by estimated impact:

| Rule | Trigger | Impact |
|------|---------|--------|
| Metal GPU not active | Apple Silicon + tok/s below Metal baseline | HIGH |
| Cold start | load_duration > 1s | MED |
| Slow prefill | prefill > 2s for prompt size | MED |
| Slow generation | tok/s below threshold for hardware | MED |
| CPU-only inference | No GPU detected | LOW |

Rules are hardware-aware — advice for Apple Silicon differs from NVIDIA differs from CPU-only. Every recommendation includes an exact action, not vague guidance.

---

## Roadmap

- [ ] llama.cpp backend (direct subprocess + output parsing)
- [ ] Gemma CPP backend
- [ ] Multi-model comparison (`evalkit compare --models a,b --prompt "..."`)
- [ ] OpenAI / Anthropic API backends
- [ ] JSON report export (for CI/CD integration)
- [ ] Latency threshold alerts (`--fail-if-ttft-above 2.0`)
- [ ] vLLM + TGI serving framework support
- [ ] HTML report output

---

## Project structure

```
evalkit/
├── evalkit/
│   ├── core/
│   │   ├── metrics.py          # InferenceResult + BenchmarkResult data models
│   │   ├── hardware.py         # Hardware detection (Apple Silicon, CUDA, RAM)
│   │   └── recommendations.py  # Recommendation engine + rules
│   ├── backends/
│   │   └── ollama.py           # Ollama REST API adapter
│   ├── reporters/
│   │   └── terminal.py         # Rich terminal output
│   └── cli/
│       └── main.py             # evalkit profile + evalkit benchmark commands
└── pyproject.toml
```

---

## Contributing

EvalKit is early and actively being built. If you run it and something's wrong or missing, open an issue. If you want to add a backend adapter or recommendation rule, PRs are welcome — the architecture is designed to make both straightforward.

---

## License

MIT
