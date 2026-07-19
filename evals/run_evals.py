"""Run the agent against all test cases and score results."""

import json
import os
import sys
import time

# Ensure we can import from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv(override=True)

from openai import OpenAI
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown
from rich import box

from agent import run_agent, MODEL, client as agent_client
from utils.tracer import Tracer
from utils.retry import retry_with_backoff

console = Console()

# ----------------------------------------------------------------
# Judge — ask the model to grade an answer against criteria
# ----------------------------------------------------------------


def judge_result(goal: str, criteria: str, answer: str) -> dict:
    """Ask the model whether *answer* satisfies *criteria* for *goal*.

    Returns {"pass": bool, "reason": str}.
    """
    judge_prompt = (
        "You are a strict but fair evaluator. Given a lead qualification goal, "
        "a set of grading criteria, and the agent's answer, determine "
        "whether the answer satisfies the criteria.\n\n"
        "Respond ONLY in JSON — no markdown, no explanation outside the "
        'JSON. Use this exact shape:\n'
        '{"pass": true/false, "reason": "brief explanation of your verdict"}'
    )

    user_msg = (
        f"Goal: {goal}\n\n"
        f"Criteria:\n{criteria}\n\n"
        f"Agent's answer:\n{answer}"
    )

    response = retry_with_backoff(
        lambda: agent_client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": judge_prompt},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
        ),
    )

    if isinstance(response, str):
        return {"pass": False, "reason": f"Judge LLM error: {response}"}

    content = response.choices[0].message.content or ""
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[-1]
        content = content.rsplit("```", 1)[0].strip()

    try:
        result = json.loads(content)
        if not isinstance(result, dict) or "pass" not in result:
            raise ValueError("Missing 'pass' key")
        return result
    except (json.JSONDecodeError, ValueError) as e:
        return {"pass": False, "reason": f"Judge parse error: {e}"}


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------


def main():
    # Load test cases
    cases_path = os.path.join(os.path.dirname(__file__), "test_cases.json")
    with open(cases_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    console.print(
        Panel(
            f"[bold cyan]Loaded {len(cases)} test cases[/]",
            border_style="cyan",
        )
    )
    console.print()

    # Set auto-approve so email tests don't block
    os.environ["AUTO_APPROVE_EMAIL"] = "y"

    results = []
    total = len(cases)

    for i, case in enumerate(cases, 1):
        cid = case["id"]
        goal = case["goal"]
        criteria = case["criteria"]

        # Header
        short = goal[:60] + ("..." if len(goal) > 60 else "")
        console.rule(f"[bold]Test {cid}/{total}: {short}[/]")

        # Run the agent
        try:
            tracer = Tracer(goal)
            answer = run_agent(goal, tracer=tracer)
            tracer.close()
        except Exception as e:
            console.print(f"[red]Agent crashed on test {cid}: {e}[/]")
            results.append(
                {
                    "id": cid,
                    "goal": goal,
                    "pass": False,
                    "reason": f"Agent crashed: {e}",
                }
            )
            console.print()
            continue

        # Judge
        console.print("[dim]Judging answer ...[/]")
        verdict = judge_result(goal, criteria, answer)
        passed = verdict.get("pass", False)
        reason = verdict.get("reason", "No reason given")

        results.append(
            {
                "id": cid,
                "goal": goal,
                "pass": passed,
                "reason": reason,
            }
        )

        tag = "[green]PASS[/]" if passed else "[red]FAIL[/]"
        console.print(f"{tag} {reason[:100]}")
        console.print()

    # ------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------
    passed_count = sum(1 for r in results if r["pass"])
    score_pct = round(passed_count / total * 100)

    table = Table(
        title=f"[bold]Eval Results: {passed_count}/{total} passed ({score_pct}%)[/]",
        box=box.ROUNDED,
        border_style="blue",
    )
    table.add_column("ID", style="dim", width=4)
    table.add_column("Goal", style="bold white", width=50)
    table.add_column("Result", width=8)
    table.add_column("Reason", width=60)

    for r in results:
        label = "[green]PASS[/]" if r["pass"] else "[red]FAIL[/]"
        short_goal = r["goal"][:47] + "..." if len(r["goal"]) > 47 else r["goal"]
        short_reason = r["reason"][:57] + "..." if len(r["reason"]) > 57 else r["reason"]
        table.add_row(str(r["id"]), short_goal, label, short_reason)

    console.print()
    console.print(table)
    console.print()

    # Overall score panel
    color = "green" if score_pct >= 70 else "yellow" if score_pct >= 40 else "red"
    console.print(
        Panel(
            f"[bold {color}]{passed_count}/{total} passed ({score_pct}%)[/]",
            border_style=color,
            title="[bold]Overall Score[/]",
        )
    )


if __name__ == "__main__":
    main()
