"""Trace viewer -- pretty-print a JSONL run log as a styled timeline."""

import json
import os
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.text import Text
from rich import box

console = Console()


def _events(path: str):
    """Yield parsed JSON events from a JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _heading(label: str, style: str = "bold cyan"):
    console.print()
    console.print(Text(f" {label} ", style=style), width=console.width)


def _ts(entry: dict) -> str:
    """Return a short timestamp for an entry."""
    raw = entry.get("timestamp", "")
    # Trim milliseconds or use the later portion
    if "." in raw:
        raw = raw.split("T")[1][:12]
    else:
        raw = raw.split("T")[1][:8]
    return raw


def _preview(text: str, max_len: int = 300) -> str:
    """Shorten a string for display."""
    if len(text) > max_len:
        return text[:max_len] + " ..."
    return text


def view_trace(path: str) -> None:
    """Pretty-print a JSONL trace file."""
    entries = list(_events(path))
    if not entries:
        console.print("[red]No events found in that file.[/]")
        return

    console.print(
        Panel(
            f"[bold]Trace file:[/] {path}\n"
            f"[bold]Events:[/] {len(entries)}",
            border_style="blue",
            title="[bold blue]Run Timeline[/]",
        )
    )

    for i, entry in enumerate(entries):
        event = entry.get("event", "?")
        ts = _ts(entry)

        if event == "goal":
            console.print(
                Panel(
                    entry.get("goal", ""),
                    border_style="blue",
                    title=f"[bold blue]Goal @ {ts}[/]",
                )
            )

        elif event == "plan":
            steps = entry.get("steps", [])
            table = Table(
                box=box.SIMPLE,
                border_style="cyan",
                title=f"Plan @ {ts}",
            )
            table.add_column("#", style="dim", width=4)
            table.add_column("Step", style="bold white")
            for s in steps:
                table.add_row(str(s.get("step", "?")), s.get("action", ""))
            console.print(table)

        elif event == "llm_call":
            purpose = entry.get("purpose", "llm")
            inp = entry.get("tokens_input")
            out = entry.get("tokens_output")
            lat = entry.get("latency_sec", "?")
            tok_str = ""
            if inp is not None and out is not None:
                tok_str = f" | {inp} in / {out} out tokens"
            console.print(
                Panel(
                    f"[bold]Latency:[/] {lat}s{tok_str}",
                    border_style="green",
                    title=f"[green]LLM call ({purpose}) @ {ts}[/]",
                )
            )

        elif event == "tool_call":
            tool = entry.get("tool", "?")
            args = entry.get("args", {})
            dur = entry.get("duration_sec", "?")
            result = entry.get("result_preview", "")

            args_str = json.dumps(args, ensure_ascii=False)
            preview = _preview(result)

            body = (
                f"[bold yellow]Tool:[/] {tool}\n"
                f"[bold yellow]Args:[/] {args_str}\n"
                f"[bold yellow]Duration:[/] {dur}s\n\n"
                f"{preview}"
            )
            console.print(
                Panel(
                    body,
                    border_style="yellow",
                    title=f"Tool call @ {ts}",
                    highlight=False,
                )
            )

        elif event == "answer":
            answer = entry.get("answer", "")
            console.print(
                Panel(
                    Markdown(answer),
                    border_style="green",
                    title=f"[bold green]Answer @ {ts}[/]",
                )
            )

        elif event == "critique":
            verdict = entry.get("verdict", {})
            passed = verdict.get("pass", False)
            reason = verdict.get("reason", "")
            fix = verdict.get("fix", "")
            style = "green" if passed else "yellow"
            label = "Critique PASSED" if passed else "Critique FAILED"
            body = reason
            if fix:
                body += f"\n\n[dim]Suggested improvement:[/] {fix}"
            console.print(
                Panel(body, border_style=style, title=f"[{style}]{label} @ {ts}[/]")
            )

        elif event == "retry":
            attempt = entry.get("attempt", "?")
            max_r = entry.get("max_retries", "?")
            error = entry.get("error", "")
            console.print(
                Panel(
                    f"Attempt {attempt}/{max_r} failed: {error}",
                    border_style="yellow",
                    title=f"Retry @ {ts}",
                )
            )

        elif event == "error":
            console.print(
                Panel(
                    entry.get("error", ""),
                    border_style="red",
                    title=f"[red]Error @ {ts}[/]",
                )
            )

        else:
            console.print(
                Panel(
                    json.dumps(entry, indent=2, ensure_ascii=False),
                    border_style="dim",
                    title=f"{event} @ {ts}",
                )
            )

    console.print(
        f"\n[dim]--- End of trace ({len(entries)} events) ---[/]"
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Find the most recent log
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        if not os.path.isdir(log_dir):
            console.print("[red]No logs/ directory found. Run the agent first.[/]")
            sys.exit(1)
        files = sorted(
            [
                f
                for f in os.listdir(log_dir)
                if f.startswith("run_") and f.endswith(".jsonl")
            ],
            reverse=True,
        )
        if not files:
            console.print("[red]No run_*.jsonl files found in logs/[/]")
            sys.exit(1)
        path = os.path.join(log_dir, files[0])
        console.print(f"[dim]Using latest trace: {path}[/]\n")
    else:
        path = sys.argv[1]

    view_trace(path)
