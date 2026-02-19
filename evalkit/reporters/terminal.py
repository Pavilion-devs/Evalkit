"""
Terminal reporter for EvalKit.

Takes an InferenceResult or BenchmarkResult and renders it to the terminal
using Rich — colors, bars, panels, the works.

Design rules:
  - Bars are relative to the max value in their group (not absolute)
  - Labels (FAST/OK/SLOW) so you don't need to memorize thresholds
  - Cold starts are flagged explicitly — they're misleading otherwise
  - Warm TTFT (prefill only) is always shown separately from cold TTFT
"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.rule import Rule
from rich import box

from evalkit.core.metrics import InferenceResult, BenchmarkResult
from evalkit.core.recommendations import Recommendation

console = Console()

# ── Thresholds ──────────────────────────────────────────────────────────────
# These define what counts as FAST / OK / SLOW.
# They're intentionally conservative — tuned for CPU and Apple Silicon.
# We'll make these configurable later.

TTFT_THRESHOLDS = {
    "fast": 0.5,   # under 0.5s → FAST (green)
    "ok":   2.0,   # under 2.0s → OK   (yellow)
                   # over  2.0s → SLOW (red)
}

TPS_THRESHOLDS = {
    "fast": 25.0,  # over 25 tok/s  → FAST (green)
    "ok":   10.0,  # over 10 tok/s  → OK   (yellow)
                   # under 10 tok/s → SLOW (red)
}

COLD_START_THRESHOLD = 0.5  # load_duration above this = flag as cold start


# ── Helpers ──────────────────────────────────────────────────────────────────

def _bar(value: float, max_value: float, width: int = 14) -> str:
    """
    Render a unicode progress bar.

    value and max_value must be in the same unit (seconds, tok/s, etc).
    We clamp to [0, 1] so values slightly over max don't break the bar.
    """
    if max_value <= 0:
        return "░" * width
    ratio = min(value / max_value, 1.0)
    filled = round(ratio * width)
    return "█" * filled + "░" * (width - filled)


def _ttft_label(ttft: float) -> Text:
    """Return a colored FAST/OK/SLOW label for a TTFT value."""
    if ttft <= TTFT_THRESHOLDS["fast"]:
        return Text("FAST", style="bold green")
    elif ttft <= TTFT_THRESHOLDS["ok"]:
        return Text("OK", style="bold yellow")
    else:
        return Text("SLOW", style="bold red")


def _tps_label(tps: float) -> Text:
    """Return a colored FAST/OK/SLOW label for a tokens/sec value."""
    if tps >= TPS_THRESHOLDS["fast"]:
        return Text("FAST", style="bold green")
    elif tps >= TPS_THRESHOLDS["ok"]:
        return Text("OK", style="bold yellow")
    else:
        return Text("SLOW", style="bold red")


def _is_cold(result: InferenceResult) -> bool:
    return result.load_duration > COLD_START_THRESHOLD


def _truncate(text: str, max_len: int = 72) -> str:
    return text if len(text) <= max_len else text[:max_len - 3] + "..."


# ── Main reporter ─────────────────────────────────────────────────────────────

def show_result(result: InferenceResult) -> None:
    """
    Render a single InferenceResult to the terminal.
    This is what runs after a single `evalkit profile` call.
    """
    cold = _is_cold(result)

    # ── Header panel ──────────────────────────────────────────────────────────
    header = Table.grid(padding=(0, 1))
    header.add_column(style="dim")
    header.add_column()

    header.add_row("Model:",    f"[bold]{result.model}[/bold]  [dim]({result.backend})[/dim]")
    header.add_row("Prompt:",   f'[italic]"{_truncate(result.prompt)}"[/italic]')
    header.add_row("Response:", f'[green]"{_truncate(result.response)}"[/green]')

    console.print()
    console.print(Panel(header, title="[bold cyan]EvalKit Profile[/bold cyan]", box=box.ROUNDED))

    # ── Timing section ────────────────────────────────────────────────────────
    # Bar scale: max of the three durations so bars are relative to each other.
    max_time = max(result.load_duration, result.prompt_eval_duration, result.eval_duration, 0.001)

    console.print()
    console.print(Rule("[bold]Timing[/bold]", style="dim"))
    console.print()

    timing_table = Table.grid(padding=(0, 2))
    timing_table.add_column(width=12)  # label
    timing_table.add_column(width=9)   # value
    timing_table.add_column(width=16)  # bar
    timing_table.add_column()          # annotation

    # Load row — only shown if there's meaningful load time
    if result.load_duration > 0.001:
        load_bar = _bar(result.load_duration, max_time)
        load_annotation = Text("COLD START", style="yellow") if cold else Text("warm", style="dim")
        timing_table.add_row(
            "Load",
            f"{result.load_duration:.3f}s",
            f"[yellow]{load_bar}[/yellow]",
            load_annotation,
        )

    # Prefill row
    prefill_bar = _bar(result.prompt_eval_duration, max_time)
    timing_table.add_row(
        "Prefill",
        f"{result.prompt_eval_duration:.3f}s",
        f"[cyan]{prefill_bar}[/cyan]",
        Text(f"{result.prompt_tokens} tokens in", style="dim"),
    )

    # Generation row
    gen_bar = _bar(result.eval_duration, max_time)
    timing_table.add_row(
        "Generation",
        f"{result.eval_duration:.3f}s",
        f"[magenta]{gen_bar}[/magenta]",
        Text(f"{result.completion_tokens} tokens out", style="dim"),
    )

    console.print(timing_table)
    console.print()

    # TTFT summary row
    ttft_label = _ttft_label(result.time_to_first_token)
    console.print(
        f"  [dim]TTFT[/dim]    [bold]{result.time_to_first_token:.3f}s[/bold]"
        f"  {ttft_label.markup}"
        + ("  [dim](includes cold load)[/dim]" if cold else "")
    )

    # If cold, also show the warm TTFT (just prefill, no load)
    # This is more meaningful for repeated use.
    if cold:
        warm_ttft = result.prompt_eval_duration
        warm_label = _ttft_label(warm_ttft)
        console.print(
            f"  [dim]TTFT (warm est.)[/dim]  [bold]{warm_ttft:.3f}s[/bold]"
            f"  {warm_label.markup}  [dim](prefill only, no load)[/dim]"
        )

    # ── Throughput section ────────────────────────────────────────────────────
    console.print()
    console.print(Rule("[bold]Throughput[/bold]", style="dim"))
    console.print()

    # Bar scale: max of gen and prefill TPS
    max_tps = max(result.tokens_per_second, result.prompt_tokens_per_second, 0.001)

    tps_table = Table.grid(padding=(0, 2))
    tps_table.add_column(width=12)
    tps_table.add_column(width=13)
    tps_table.add_column(width=16)
    tps_table.add_column()

    # Generation TPS
    gen_tps_bar = _bar(result.tokens_per_second, max_tps)
    gen_tps_label = _tps_label(result.tokens_per_second)
    tps_table.add_row(
        "Generation",
        f"{result.tokens_per_second:.1f} tok/s",
        f"[magenta]{gen_tps_bar}[/magenta]",
        gen_tps_label,
    )

    # Prefill TPS (almost always much faster than generation)
    prefill_tps_bar = _bar(result.prompt_tokens_per_second, max_tps)
    prefill_tps_label = _tps_label(result.prompt_tokens_per_second)
    tps_table.add_row(
        "Prefill",
        f"{result.prompt_tokens_per_second:.1f} tok/s",
        f"[cyan]{prefill_tps_bar}[/cyan]",
        prefill_tps_label,
    )

    console.print(tps_table)
    console.print()


def show_recommendations(recommendations: list[Recommendation]) -> None:
    """
    Render a ranked list of recommendations to the terminal.

    Each recommendation shows:
      - Impact badge (HIGH / MED / LOW) with color
      - Title
      - Description of what's happening
      - Exact action to take
      - Estimated speedup if we have one
    """
    if not recommendations:
        console.print(Rule("[bold]Recommendations[/bold]", style="dim"))
        console.print()
        console.print("  [green]No issues detected. Inference looks healthy.[/green]")
        console.print()
        return

    console.print(Rule("[bold]Recommendations[/bold]", style="dim"))
    console.print()

    impact_styles = {
        "high":   ("HIGH", "bold red"),
        "medium": ("MED",  "bold yellow"),
        "low":    ("LOW",  "dim"),
    }

    for rec in recommendations:
        badge_text, badge_style = impact_styles[rec.impact]

        # Each recommendation is a small grid: badge | content
        rec_grid = Table.grid(padding=(0, 1))
        rec_grid.add_column(width=7, no_wrap=True)   # badge
        rec_grid.add_column()                         # content (wraps naturally)

        # Build the content block as one Text object so Rich handles wrapping
        content = Text()
        content.append(rec.title + "\n", style="bold")
        content.append(rec.description + "\n", style="dim")

        if rec.estimated_speedup:
            content.append(f"Estimated gain: {rec.estimated_speedup}\n", style="green")

        content.append("→ What to do:\n", style="cyan")
        for line in rec.action.split("\n"):
            content.append(f"  {line}\n", style="")

        badge = Text(f"[{badge_text}]", style=badge_style, justify="left")
        rec_grid.add_row(badge, content)

        console.print(rec_grid)
        console.print()


def show_benchmark(benchmark: BenchmarkResult) -> None:
    """
    Render aggregated BenchmarkResult (multi-run) to the terminal.
    Shows mean, p50, p95, p99 across all iterations.
    """
    if not benchmark.results:
        console.print("[red]No results to display.[/red]")
        return

    first = benchmark.results[0]
    n = len(benchmark.results)

    # ── Header ────────────────────────────────────────────────────────────────
    header = Table.grid(padding=(0, 1))
    header.add_column(style="dim")
    header.add_column()
    header.add_row("Model:",      f"[bold]{first.model}[/bold]  [dim]({first.backend})[/dim]")
    header.add_row("Prompt:",     f'[italic]"{_truncate(first.prompt)}"[/italic]')
    header.add_row("Iterations:", f"[bold]{n}[/bold] runs")

    console.print()
    console.print(Panel(header, title="[bold cyan]EvalKit Benchmark[/bold cyan]", box=box.ROUNDED))

    # ── Latency distribution ──────────────────────────────────────────────────
    console.print()
    console.print(Rule("[bold]Latency Distribution (total)[/bold]", style="dim"))
    console.print()

    # Scale bars to the p99 value
    max_latency = benchmark.p99_total_duration or 0.001

    lat_table = Table.grid(padding=(0, 2))
    lat_table.add_column(width=8)
    lat_table.add_column(width=9)
    lat_table.add_column(width=16)
    lat_table.add_column()

    for label, value in [
        ("Mean", benchmark.mean_total_duration),
        ("p50",  benchmark.p50_total_duration),
        ("p95",  benchmark.p95_total_duration),
        ("p99",  benchmark.p99_total_duration),
    ]:
        bar = _bar(value, max_latency)
        # p95 and p99 get more scrutiny — highlight if they're much worse than mean
        if label in ("p95", "p99") and value > benchmark.mean_total_duration * 1.5:
            style = "red"
        else:
            style = "cyan"
        lat_table.add_row(label, f"{value:.3f}s", f"[{style}]{bar}[/{style}]", "")

    console.print(lat_table)

    # ── Throughput summary ────────────────────────────────────────────────────
    console.print()
    console.print(Rule("[bold]Throughput[/bold]", style="dim"))
    console.print()

    tps_label = _tps_label(benchmark.mean_tokens_per_second)
    ttft_label = _ttft_label(benchmark.mean_ttft)

    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="dim", width=22)
    summary.add_column()

    summary.add_row("Mean tok/s (gen):",  f"[bold]{benchmark.mean_tokens_per_second:.1f}[/bold]  {tps_label.markup}")
    summary.add_row("Mean TTFT:",         f"[bold]{benchmark.mean_ttft:.3f}s[/bold]  {ttft_label.markup}")

    console.print(summary)
    console.print()
