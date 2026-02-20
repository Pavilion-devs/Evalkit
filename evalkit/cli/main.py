"""
EvalKit CLI entry point.

Commands:
  evalkit profile        — profile a single ollama inference run
  evalkit benchmark      — run N iterations on ollama and show aggregate stats
  evalkit profile-llama  — profile a single llama.cpp inference run
"""

import typer
from rich.console import Console

from evalkit.backends.ollama import OllamaBackend
from evalkit.backends.llamacpp import LlamaCppBackend
from evalkit.core.metrics import BenchmarkResult
from evalkit.core.hardware import detect_hardware
from evalkit.core.recommendations import RecommendationEngine
from evalkit.reporters.terminal import show_result, show_benchmark, show_recommendations

app = typer.Typer(
    name="evalkit",
    help="LLM inference profiler. Tells you why your model is slow and how to fix it.",
    add_completion=False,
)
console = Console()


def _get_ollama_backend() -> OllamaBackend:
    backend = OllamaBackend()
    if not backend.is_available():
        console.print("[red]Cannot connect to ollama. Is the ollama app running?[/red]")
        raise typer.Exit(code=1)
    return backend


@app.command()
def profile(
    model: str = typer.Option("llama3.2:3b", "--model", "-m", help="Model name (must be pulled in ollama)"),
    prompt: str = typer.Option(..., "--prompt", "-p", help="The prompt to run"),
):
    """
    Profile a single inference run and display timing + throughput metrics.
    """
    backend = _get_ollama_backend()

    hardware = detect_hardware()
    console.print(f"\n[dim]Running inference on [bold]{model}[/bold]...[/dim]")
    console.print(f"[dim]Hardware: {hardware.summary()}[/dim]")

    result = backend.run(model=model, prompt=prompt)
    show_result(result)

    recs = RecommendationEngine().analyze(result, hardware)
    show_recommendations(recs)


@app.command()
def benchmark(
    model: str = typer.Option("llama3.2:3b", "--model", "-m", help="Model name"),
    prompt: str = typer.Option(..., "--prompt", "-p", help="The prompt to run"),
    iterations: int = typer.Option(5, "--iterations", "-n", help="Number of times to run the prompt"),
):
    """
    Run the same prompt N times and show aggregate latency distribution.

    The first run is often a cold start (model loading from disk).
    Subsequent runs reflect real warm performance.
    """
    backend = _get_ollama_backend()

    hardware = detect_hardware()
    console.print(f"\n[dim]Running [bold]{iterations}[/bold] iterations on [bold]{model}[/bold]...[/dim]")
    console.print(f"[dim]Hardware: {hardware.summary()}[/dim]")

    results = []
    for i in range(1, iterations + 1):
        console.print(f"  [dim]iteration {i}/{iterations}[/dim]", end="\r")
        result = backend.run(model=model, prompt=prompt, iteration=i)
        results.append(result)

    console.print(" " * 40, end="\r")  # clear the iteration line

    benchmark_result = BenchmarkResult(results=results)
    show_benchmark(benchmark_result)

    # Analyze the last warm result for recommendations
    # (skip first run if it was a cold start — cold runs skew the analysis)
    warm_results = [r for r in results if r.load_duration < 1.0]
    analysis_result = warm_results[-1] if warm_results else results[-1]
    recs = RecommendationEngine().analyze(analysis_result, hardware)
    show_recommendations(recs)


@app.command(name="profile-llama")
def profile_llama(
    model: str = typer.Option(..., "--model", "-m", help="Path to a .gguf model file"),
    prompt: str = typer.Option(..., "--prompt", "-p", help="The prompt to run"),
    ctx_size: int = typer.Option(4096, "--ctx-size", "-c", help="KV cache context window in tokens"),
    threads: int = typer.Option(None, "--threads", "-t", help="CPU threads (default: half logical cores)"),
    n_gpu_layers: int = typer.Option(None, "--n-gpu-layers", "-ngl", help="GPU layers to offload (0 = CPU only)"),
):
    """
    Profile a single llama.cpp inference run using the llama-completion binary.

    Model must be a path to a .gguf file. You can use ollama's cached blobs directly:
    ~/.ollama/models/blobs/sha256-<hash>
    """
    try:
        backend = LlamaCppBackend(
            context_size=ctx_size,
            threads=threads,
            n_gpu_layers=n_gpu_layers,
        )
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1)

    if not backend.is_available():
        console.print("[red]llama-completion binary not found or not executable.[/red]")
        console.print("[dim]Install with: brew install llama.cpp[/dim]")
        raise typer.Exit(code=1)

    hardware = detect_hardware()
    console.print(f"\n[dim]Running llama.cpp inference on [bold]{model}[/bold]...[/dim]")
    console.print(f"[dim]Hardware: {hardware.summary()}[/dim]")

    try:
        result = backend.run(model_path=model, prompt=prompt)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1)
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(code=1)

    show_result(result)

    recs = RecommendationEngine().analyze(result, hardware)
    show_recommendations(recs)
