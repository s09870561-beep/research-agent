"""
research-agent  ·  demo.py

Polished single-file entry point for interviews.
Researches a lead (company/person), produces a qualification
summary and optional outreach email, with full terminal output.

Usage:
    python demo.py
"""

import os
import json
import re
import time
import textwrap
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

# ── Rich terminal output ──────────────────────────────────────────────
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich import box

console = Console()

# ── Config ────────────────────────────────────────────────────────────
MODEL = "deepseek-v4-flash-free"

load_dotenv(override=True)
OPENCODE_ZEN_API_KEY = os.getenv("OPENCODE_ZEN_API_KEY", "").strip()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()
AUTO_APPROVE_EMAIL = os.getenv("AUTO_APPROVE_EMAIL", "").strip().lower() == "y"

if not OPENCODE_ZEN_API_KEY:
    console.print("[red]Error:[/red] OPENCODE_ZEN_API_KEY not set in .env")
    console.print("Create a .env file with:\n  OPENCODE_ZEN_API_KEY=sk-...")
    exit(1)
if not TAVILY_API_KEY:
    console.print("[red]Error:[/red] TAVILY_API_KEY not set in .env")
    console.print("Create a .env file with:\n  TAVILY_API_KEY=tvly-...")
    exit(1)

client = OpenAI(
    base_url="https://opencode.ai/zen/v1",
    api_key=OPENCODE_ZEN_API_KEY,
)


# ── Utilities ─────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    """Strip characters the terminal can't print."""
    return text.encode(console.encoding or "utf-8", errors="replace").decode(
        console.encoding or "utf-8"
    )


def now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def llm(messages, purpose="chat", temperature=0.7, max_tokens=8192):
    """Call the LLM and return content + token usage."""
    start = time.time()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    elapsed = round(time.time() - start, 2)
    usage = resp.usage
    line = (
        f"  [dim]{now()}[/dim]  {purpose}  "
        f"in={usage.prompt_tokens}  out={usage.completion_tokens}  "
        f"latency={elapsed}s"
    )
    console.print(line)
    return resp.choices[0].message.content or "", usage


def retry(messages, purpose="chat", max_retries=3, **kwargs):
    """Call LLM with exponential backoff."""
    for attempt in range(1, max_retries + 1):
        try:
            content, _ = llm(messages, purpose=purpose, **kwargs)
            return content
        except Exception as e:
            if attempt < max_retries:
                wait = 2 ** (attempt - 1)
                console.print(
                    f"  [yellow]Retry {attempt}/{max_retries}[/yellow]  "
                    f"waiting {wait}s  error: {e}"
                )
                time.sleep(wait)
            else:
                console.print(
                    f"  [red]Failed after {max_retries} retries:[/red] {e}"
                )
                return ""
    return ""


# ── Tools ─────────────────────────────────────────────────────────────

def web_search(query: str) -> str:
    """Search the web via Tavily."""
    try:
        from tavily import TavilyClient
        tavily = TavilyClient(api_key=TAVILY_API_KEY)
        resp = tavily.search(query=query, max_results=4)
        results = resp.get("results", [])
        if not results:
            return "No results found."
        lines = []
        for r in results:
            title = r.get("title", "Untitled")
            url = r.get("url", "")
            content = r.get("content", "")
            snippet = _clean(content[:300])
            lines.append(f"- [{title}]({url})\n  {snippet}")
        return "\n\n".join(lines)
    except Exception as e:
        return f"Search error: {e}"


def send_email(to: str, subject: str, body: str) -> str:
    """Simulate sending an email after user approval."""
    console.print()
    console.print(Panel(
        f"To:      {to}\n"
        f"Subject: {subject}\n"
        f"Body:\n{textwrap.indent(body, '  ')}",
        title="[yellow]Email Approval Required[/yellow]",
        border_style="yellow",
    ))
    if AUTO_APPROVE_EMAIL:
        console.print("  [dim](auto-approved via AUTO_APPROVE_EMAIL=y)[/dim]")
        approved = True
    else:
        resp = console.input("  Send this email? [y/N] ").strip().lower()
        approved = resp == "y"
    if approved:
        console.print(f"  [green]Email sent to {to}[/green]")
        return f"Email sent successfully to {to} with subject '{subject}'."
    else:
        console.print("  [red]Email rejected by user.[/red]")
        return "Email was rejected by the user. Do not retry."


# ── Agent entry point ────────────────────────────────────────────────

def run_demo():
    """Run one polished lead qualification end to end."""
    DEFAULT_GOAL = (
        "Research the company Anthropic. Determine if they're a "
        "good fit for our enterprise AI consulting service, and "
        "draft a qualification summary including a recommended "
        "outreach email to their head of partnerships."
    )

    goal = Prompt.ask(
        "[bold cyan]Which lead should I research? (company or person)[/bold cyan]",
        default="",
    )
    if not goal.strip():
        goal = DEFAULT_GOAL
        console.print(f"  [dim](Using default question)[/dim]")

    console.print()
    console.print(Panel(
        _clean(goal),
        title="[bold cyan]Lead Research Goal[/bold cyan]",
        border_style="cyan",
    ))

    # ── Step 1: Plan ──────────────────────────────────────────────────
    console.print(f"\n[bold]Step 1: Plan[/bold]  [{now()}]")

    plan_prompt = f"""You are a lead qualification planner. Break the following goal into 2-4 concrete steps.
Return ONLY a JSON array of strings, one per step. No commentary.

Goal: {goal}"""
    plan_text = retry(
        [{"role": "user", "content": plan_prompt}],
        purpose="plan",
        max_retries=2,
    )

    try:
        steps = json.loads(plan_text)
        if isinstance(steps, list):
            table = Table(box=box.ROUNDED, border_style="blue")
            table.add_column("#", style="dim", width=3)
            table.add_column("Step")
            for i, s in enumerate(steps, 1):
                table.add_row(str(i), _clean(s))
            console.print(table)
        else:
            steps = [goal]
    except (json.JSONDecodeError, TypeError):
        steps = [goal]

    # ── Step 2: Research loop ─────────────────────────────────────────
    console.print(f"\n[bold]Step 2: Research[/bold]  [{now()}]")

    messages = [
        {
            "role": "system",
            "content": (
                "You are a lead research and outreach assistant. Answer the user's goal using the tools available. "
                "When you need information about a company or person, call web_search. "
                "When research is complete, produce a qualification summary covering: company overview, "
                "potential product fit, recommended outreach angle, and next steps. "
                "If the user asks for an outreach email, draft it and use send_email.\n\n"
                "Keep answers factual and cite sources. "
                "When you have enough information, produce the final answer and stop.\n\n"
                "Available tools:\n"
                "- web_search(query: str)  -- search the web for company/lead info\n"
                "- send_email(to, subject, body)  -- send an outreach email (requires approval)\n\n"
                "You MUST respond in exactly one of these two formats:\n\n"
                "1. To call a tool:\n"
                'TOOL: web_search\n'
                'ARGS: {"query": "what you want to search"}\n\n'
                "2. When you have enough information and are ready to give the final answer:\n"
                "ANSWER: your final answer here, with full details\n\n"
                "Examples:\n\n"
                'User: Research Acme Corp and determine if they are a good lead.\n'
                'Assistant: TOOL: web_search\n'
                'ARGS: {"query": "Acme Corp business model funding 2026"}\n\n'
                "(after receiving search results)\n"
                "Assistant: ANSWER: Lead Qualification Summary for Acme Corp -- ...\n\n"
                "Always start your response with either TOOL: or ANSWER: -- no greetings, "
                "no explanations, no thinking aloud before the prefix. "
                "Just the prefix and then the content."
            ),
        },
        {"role": "user", "content": goal},
    ]

    max_turns = 6
    final_answer = ""
    last_response_text = ""

    for turn in range(1, max_turns + 1):
        response_text = retry(messages, purpose=f"turn {turn}", max_retries=2)
        if response_text:
            last_response_text = response_text

        # Check for ANSWER: anywhere in the response
        answer_match = re.search(r'\bANSWER\s*:\s*(.*)', response_text, re.DOTALL)
        # Check for TOOL: anywhere in the response (must be on a new line or at start)
        tool_match = re.search(r'(?:^|\n)\s*TOOL\s*:\s*(\w+)', response_text)
        args_match = re.search(r'(?:^|\n)\s*ARGS\s*:\s*(\{.+\})', response_text, re.DOTALL)

        if answer_match:
            final_answer = answer_match.group(1).strip()
            console.print(f"\n  [green]Answer received after {turn} turn(s)[/green]")
            break

        if tool_match:
            tool_name = tool_match.group(1).strip()
            args = {}
            if args_match:
                try:
                    args = json.loads(args_match.group(1))
                except (json.JSONDecodeError, TypeError):
                    args = {}

            console.print()
            console.print(f"  [blue]Tool:[/blue] {tool_name}")
            if args:
                console.print(f"  Args: {json.dumps(args, ensure_ascii=False)}")

            if tool_name == "web_search":
                query = args.get("query", "")
                result = web_search(query)
                preview = _clean(result[:200])
                console.print(f"  Result: {preview}...")
                messages.append({"role": "user", "content": f"web_search result for '{query}':\n{result}"})

            elif tool_name == "send_email":
                result = send_email(
                    args.get("to", ""),
                    args.get("subject", ""),
                    args.get("body", ""),
                )
                messages.append({"role": "user", "content": f"send_email result: {result}"})

            else:
                messages.append({"role": "user", "content": f"Unknown tool '{tool_name}'."})

            # Add a hint to continue
            messages.append({
                "role": "user",
                "content": (
                    "Continue. If you have enough information, produce your final answer "
                    "prefixed with ANSWER:. "
                    "If you need another tool call, use TOOL:/ARGS:."
                ),
            })
        else:
            # No ANSWER: or TOOL: found anywhere in the response
            if len(response_text.strip()) < 20:
                # Trivial/empty response — keep looping, don't treat as final answer
                console.print(f"  [dim](empty response on turn {turn}, continuing)[/dim]")
                messages.append({
                    "role": "user",
                    "content": "I did not receive a valid response. Please start with TOOL: or ANSWER:.",
                })
                continue
            # Model probably answered in plain prose — treat everything as answer
            final_answer = response_text
            console.print(f"\n  [yellow]No ANSWER: prefix found, using raw response as answer[/yellow]")
            break
    else:
        # Loop exhausted max_turns without seeing ANSWER:.
        # Force a final synthesis call in plain text format.
        if not final_answer.strip():
            console.print(f"  [yellow]Max turns reached, requesting final synthesis...[/yellow]")
            synthesis = retry(
                [*messages, {
                    "role": "user",
                    "content": (
                        "You have done enough research. Now produce your final COMPLETE lead qualification "
                        "in plain text with all sections filled in (company overview, potential fit, "
                        "outreach angle, next steps). Do NOT use TOOL: or ANSWER: prefixes. "
                        "Do NOT include any thinking, planning, or discussion of how to find more information. "
                        "Just write the detailed qualification directly."
                    ),
                }],
                purpose="final synthesis",
                max_retries=2,
                temperature=0.3,
            )
            final_answer = synthesis if synthesis.strip() else (last_response_text if last_response_text.strip() else "I was unable to complete the research within the allowed turns.")

    # Fallback: if answer is still empty but the model said something, use it
    if not final_answer.strip() and last_response_text.strip():
        final_answer = last_response_text
        console.print(f"\n  [yellow]Using model's raw response as answer (no ANSWER: prefix found)[/yellow]")

    # ── Step 3: Critique ──────────────────────────────────────────────
    console.print(f"\n[bold]Step 3: Self-critique[/bold]  [{now()}]")

    critique_prompt = f"""You are a strict reviewer. Judge whether the following answer satisfies the goal.

Goal: {goal}

Answer:
{final_answer}

Return a JSON object with:
- "pass": true or false
- "reason": one-sentence explanation
- "fix": if pass is false, a short instruction on what to improve

Return ONLY valid JSON, no commentary."""
    critique_text = retry(
        [{"role": "user", "content": critique_prompt}],
        purpose="critique",
        temperature=0.2,
        max_retries=2,
    )

    try:
        verdict = json.loads(critique_text)
    except (json.JSONDecodeError, TypeError):
        verdict = {"pass": True, "reason": "Could not parse critique."}

    if verdict.get("pass"):
        console.print(Panel(
            _clean(verdict.get("reason", "Answer passed review.")),
            title="[green]Critique: PASS[/green]",
            border_style="green",
        ))
    else:
        console.print(Panel(
            _clean(verdict.get("reason", "Answer needs improvement.")),
            title="[red]Critique: FAIL[/red]",
            border_style="red",
        ))
        if verdict.get("fix"):
            console.print(f"  Fix: {_clean(verdict['fix'])}")

    # ── Step 4: Print final answer ────────────────────────────────────
    console.print(f"\n[bold]Final Answer[/bold]  [{now()}]")
    console.print()
    console.print(Panel(
        Markdown(_clean(final_answer)),
        border_style="cyan",
    ))

    console.print()
    console.print(Panel(
        "[cyan]Demo complete.[/cyan]  The agent planned, researched the lead, "
        "produced a qualification summary, and self-critiqued.  "
        "Check logs/ for detailed traces.",
        border_style="cyan",
    ))

    return final_answer


# ── Main guard ────────────────────────────────────────────────────────

if __name__ == "__main__":
    console.print("[bold cyan]Lead Research & Outreach Agent — Demo[/bold cyan]")
    console.print("=" * 50)

    start = time.time()
    result = run_demo()
    elapsed = round(time.time() - start, 2)

    console.print(f"\n[dim]Total time: {elapsed}s[/dim]")
