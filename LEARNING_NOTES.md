# EvalKit — Learning Notes

A running log of concepts covered while building this project.
Updated as we go. Use this to revise.

---

## 1. The Two Phases of Inference

Every LLM inference run has two distinct phases. They're fundamentally different operations.

### Prefill (Prompt Evaluation)
- The model reads your **entire prompt simultaneously** — all tokens at once, in parallel
- This is possible because of the transformer attention mechanism (every token can look at every other token at the same time)
- The **output of prefill is the KV Cache** — not text, but a compressed memory of what the prompt said
- Prefill time scales linearly with prompt length — longer prompt = slower prefill
- Measured as: `prompt_eval_duration` in ollama, `prompt eval time` in llama.cpp

### Generation (Decode)
- The model generates **one token at a time**, sequentially — no parallelism possible
- Each forward pass produces one token, which is fed back as input for the next pass
- This is called **autoregressive generation** — each step depends on the previous
- Measured as: `eval_duration` in ollama, `eval time` in llama.cpp

```
Prompt tokens → [Prefill: parallel] → KV Cache built
KV Cache + token 1 → predict token 2
KV Cache + token 1, 2 → predict token 3
... (one step at a time)
```

**Why streaming "feels" faster:** The computation is identical. Streaming just sends
each token to the user the moment it's generated, instead of buffering everything.

---

## 2. The KV Cache

- KV = Key-Value — named after the matrices inside transformer attention layers
- Built during prefill, stored in RAM/VRAM
- Stores the model's "memory" of the prompt so generation never re-reads it
- Without it, every generation step would reprocess the entire context from scratch
- Larger context = larger KV cache = more memory pressure
- KV cache hit rate matters in multi-turn conversations — if the cache gets evicted, the next turn pays full prefill cost again

---

## 3. TTFT — Time to First Token

**TTFT = load_duration + prompt_eval_duration**

This is the latency a user *feels* before any output appears.

- **Cold start TTFT** = model load time + prefill time (can be 10+ seconds)
- **Warm TTFT** = prefill only (model already in RAM, load ≈ 0)
- Warm TTFT is the real floor — you can't go faster than prefill
- Long system prompts directly inflate TTFT even on warm runs

---

## 4. Cold Start vs Warm Start

| | Cold Start | Warm Start |
|---|---|---|
| Model in memory? | No — loads from disk | Yes — already loaded |
| Load time | Seconds (depends on model size) | ~0ms |
| TTFT | High | Low (just prefill) |
| When it happens | First request after idle timeout | Subsequent requests |

**ollama unloads models after 5 minutes of inactivity by default.**
Fix: set `keep_alive: -1` in your request payload to keep model warm indefinitely.

---

## 5. Throughput Metrics

### Tokens per Second (tok/s)
- **Generation TPS** = `completion_tokens / eval_duration` — the main speed metric
- **Prefill TPS** = `prompt_tokens / prompt_eval_duration` — usually much faster than gen
- Prefill TPS is higher because it's parallelized; generation is sequential

### Rough TPS benchmarks by hardware
| Hardware | Expected TPS (3B model) |
|----------|------------------------|
| CPU only | 5–15 tok/s |
| Apple Silicon (Metal) | 25–60 tok/s |
| NVIDIA GPU (CUDA) | 50–200+ tok/s |

---

## 6. Percentiles — Why Mean Lies

When benchmarking, run multiple iterations and look at the **distribution**, not just the mean.

**Example — 10 runs:**
```
Sorted: [0.28, 0.28, 0.29, 0.29, 0.29, 0.30, 0.30, 0.31, 0.85, 1.20]
```
- **Mean: 0.44s** — pulled up by 2 outlier runs, not representative
- **p50: 0.29s** — half the runs were faster than this (typical user)
- **p95: ~0.85s** — 95% of runs were faster (bad day user)
- **p99: ~1.20s** — worst case (1 in 100 requests)

**Why p95/p99 matter:** On a system serving 1000 requests/day, p95 being high
means 50 users/day have a bad experience. That's noticeable.

**Minimum runs for reliable percentiles:**
- p50: 10+ runs
- p95: 20+ runs
- p99: 100+ runs

---

## 7. Quantization — Quality vs Speed vs Size

Quantization reduces the precision of model weights to save memory and increase speed.
The format is baked into the GGUF filename:

```
mistral-7b-instruct-v0.2.Q4_K_M.gguf
                            ↑
                     quantization level
```

| Level | Quality | Speed | File Size |
|-------|---------|-------|-----------|
| fp16 | Reference | Slowest | Largest |
| Q8_0 | Near-identical | Slow | Large |
| Q6_K | Excellent | Medium | Medium |
| Q5_K_M | Very good | Medium | Medium |
| **Q4_K_M** | **Good** | **Fast** | **Small** | ← sweet spot |
| Q3_K_M | OK | Faster | Smaller |
| Q2_K | Noticeable loss | Fastest | Smallest |

**Key insight:** Lighter quantization reduces memory bandwidth per token,
which directly improves generation speed — especially on CPU.

---

## 8. stdout vs stderr in llama.cpp

llama.cpp splits its output across two streams:

- **stdout** → the model's generated text (the actual response)
- **stderr** → diagnostic info, loading logs, timing data

**The timing lines we parse live on stderr:**
```
llama_print_timings:        load time =   1234.56 ms
llama_print_timings:  prompt eval time =    567.89 ms /    38 tokens
llama_print_timings:        eval time =   3456.78 ms /    99 runs
llama_print_timings:       total time =   4123.45 ms /   137 tokens
```
All durations are in **milliseconds** (ollama uses nanoseconds — different!).

---

## 9. Subprocess vs HTTP

| | ollama backend | llama.cpp backend |
|---|---|---|
| How we call it | HTTP POST to localhost | Spawn a child process |
| Response format | JSON | Raw text on stdout + stderr |
| Reliability | Versioned API | Can change between versions |
| Error handling | HTTP status codes | Exit codes + stderr messages |

**subprocess pattern:**
```python
import subprocess

process = subprocess.run(
    ["llama-cli", "-m", model_path, "-p", prompt],
    capture_output=True,   # captures both stdout and stderr
    text=True              # decodes bytes to string
)

response = process.stdout      # model output
diagnostics = process.stderr   # timing data to parse
```

---

## 10. Hardware Detection

Apple Silicon Macs use **Metal** as their GPU API (not CUDA — that's NVIDIA).
ollama automatically uses Metal on Apple Silicon. If tok/s is below ~25 on an
M-series chip, Metal may not be engaged.

**How to verify Metal is active in ollama:**
```bash
ollama ps   # shows model + GPU % next to it
```

---

## 11. The EvalKit Abstraction

All backends produce one normalized type: `InferenceResult`.
All durations in **seconds** (float), regardless of what the backend uses natively.

```
Backend (ollama / llama.cpp / OpenAI)
    ↓  translates raw output
InferenceResult (normalized, seconds)
    ↓  consumed by
Terminal Reporter + Recommendations Engine + CLI
```

Adding a new backend = write one file, produce InferenceResult.
Everything else just works.

---

*Updated as we build. Add new sections here when new concepts are introduced.*
