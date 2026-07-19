"""Lead Research & Outreach Agent -- core agent loop using OpenCode Zen."""

import json
import os
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table

from tools.search import web_search
from tools.actions import send_email as do_send_email
from memory.store import save_memory, recall_memory
from utils.retry import retry_with_backoff
from utils.tracer import Tracer

console = Console()

# -- OpenCode Zen client -------------------------------------------------

client = OpenAI(
    base_url="https://opencode.ai/zen/v1",
    api_key=os.getenv("OPENCODE_ZEN_API_KEY", "").strip(),
)

MODEL = "deepseek-v4-flash-free"

# -- Tool schema (OpenAI function-calling format) -----------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current company or lead information. "
                           "Returns a formatted string of search results with titles, URLs, and snippets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query string (e.g. 'best AI frameworks 2026').",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall_memory",
            "description": "Search past research sessions by keyword. Use this when "
                           "the user asks about something you may have researched before.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Keywords to match against past questions (e.g. 'AI frameworks solo developer').",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "IMPORTANT: This action is irreversible and sends a real outreach email. "
                           "Sends an email to a recipient with a subject line and body. "
                           "Use ONLY after the user explicitly asks you to send an outreach email.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "The recipient email address.",
                    },
                    "subject": {
                        "type": "string",
                        "description": "The email subject line.",
                    },
                    "body": {
                        "type": "string",
                        "description": "The email body content (plain text).",
                    },
                },
                "required": ["to", "subject", "body"],
            },
        },
    },
]

# -- Plan generation ----------------------------------------------------


def create_plan(user_goal: str, tracer=None) -> list[dict]:
    """Break the user's goal into 2-5 concrete research steps."""
    plan_prompt = (
        "You are a lead qualification planner. Break the user's goal into 2-5 concrete, "
        "actionable research steps. Respond ONLY with a JSON array of objects, "
        "each with keys 'step' (integer) and 'action' (string describing what "
        "to research). No markdown, no explanation, nothing else.\n\n"
        "Example:\n"
        '[{"step": 1, "action": "Search for the company\'s business model and '
        'revenue information"}, {"step": 2, "action": "Identify the company\'s '
        "technology stack and potential fit for our product\"}]\n"
    )

    t0 = time.time()
    response = retry_with_backoff(
        lambda: client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": plan_prompt},
                {"role": "user", "content": user_goal},
            ],
            temperature=0.3,
        ),
        tracer=tracer,
    )
    latency = time.time() - t0

    if isinstance(response, str):
        console.print(f"[red]{response}[/]")
        if tracer:
            tracer.log_error(f"Plan LLM call failed: {response}")
        return [
            {"step": 1, "action": f"Search for information about: {user_goal}"},
            {"step": 2, "action": "Synthesise findings into a final answer"},
        ]

    if tracer:
        usage = getattr(response, "usage", None)
        tracer.log_llm_call(
            model=MODEL,
            tokens_input=usage.prompt_tokens if usage else None,
            tokens_output=usage.completion_tokens if usage else None,
            latency_sec=latency,
            purpose="create_plan",
        )

    content = response.choices[0].message.content or ""

    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[-1]
        content = content.rsplit("```", 1)[0].strip()

    try:
        steps = json.loads(content)
        if not isinstance(steps, list):
            raise ValueError("Response was not a list")
        return steps
    except (json.JSONDecodeError, ValueError) as e:
        console.print(f"[red]Failed to parse plan: {e}[/]")
        console.print(f"[dim]Raw response: {content}[/]")
        return [
            {"step": 1, "action": f"Search for information about: {user_goal}"},
            {"step": 2, "action": "Synthesise findings into a final answer"},
        ]


def print_plan(steps: list[dict]) -> None:
    """Display the plan as a styled table."""
    table = Table(
        title="[bold cyan]Research Plan[/]",
        border_style="cyan",
        title_justify="left",
    )
    table.add_column("#", style="dim", width=4)
    table.add_column("Step", style="bold white")

    for s in steps:
        table.add_row(str(s["step"]), s["action"])

    console.print(table)
    console.print()

# -- Self-critique ------------------------------------------------------


def critique_answer(user_goal: str, final_answer: str, tracer=None) -> dict:
    """Have the model review its own answer for quality and completeness.

    Returns a dict with keys *pass* (bool), *reason* (str), *fix* (str).
    Returns a default pass=True dict if parsing fails so it doesn't
    get stuck in a critique loop.
    """
    prompt = (
        "You are a strict reviewer. Does this answer fully and accurately "
        "satisfy the user's goal? Respond ONLY in JSON — no markdown, no "
        "explanation outside the JSON. Use this exact shape:\n"
        '{"pass": true/false, "reason": "brief explanation of your '
        'verdict", "fix": "what is missing or wrong, if anything"}'
    )

    t0 = time.time()
    response = retry_with_backoff(
        lambda: client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Goal: {user_goal}\n\nAnswer:\n{final_answer}"},
            ],
            temperature=0.2,
        ),
        tracer=tracer,
    )
    latency = time.time() - t0

    if isinstance(response, str):
        console.print(f"[red]Critique failed (LLM error): {response}[/]")
        if tracer:
            tracer.log_error(f"Critique LLM call failed: {response}")
        return {"pass": True, "reason": "Critique unavailable due to LLM error", "fix": ""}

    if tracer:
        usage = getattr(response, "usage", None)
        tracer.log_llm_call(
            model=MODEL,
            tokens_input=usage.prompt_tokens if usage else None,
            tokens_output=usage.completion_tokens if usage else None,
            latency_sec=latency,
            purpose="critique",
        )

    content = response.choices[0].message.content or ""
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[-1]
        content = content.rsplit("```", 1)[0].strip()

    try:
        verdict = json.loads(content)
        if not isinstance(verdict, dict) or "pass" not in verdict:
            raise ValueError("Missing 'pass' key")
        return verdict
    except (json.JSONDecodeError, ValueError) as e:
        console.print(f"[red]Failed to parse critique: {e}[/]")
        console.print(f"[dim]Raw: {content}[/]")
        return {"pass": True, "reason": "Critique parsing failed", "fix": ""}


def print_critique(verdict: dict) -> None:
    """Display the critique result in a coloured panel."""
    passed = verdict.get("pass", False)
    style = "green" if passed else "yellow"
    label = "Critique PASSED" if passed else "Critique FAILED"
    reason = verdict.get("reason", "No reason given.")
    fix = verdict.get("fix", "")
    body = f"[bold]{reason}[/]"
    if fix:
        body += f"\n\n[dim]Suggested improvement:[/] {fix}"
    console.print(Panel(body, border_style=style, title=f"[bold {style}]{label}[/]"))


# -- Tool execution ------------------------------------------------------


def _execute_tool(tool_call, tracer=None) -> str:
    """Execute a single tool call and return the result string.

    High-risk actions (send_email) require interactive user approval.
    External calls (web_search, OpenCode Zen) are wrapped with retries.
    """
    fn = tool_call.function
    name = fn.name
    args = json.loads(fn.arguments)

    # -- Approval gate for high-risk actions ---------------------------
    if name == "send_email":
        to = args.get("to", "?")
        subject = args.get("subject", "?")
        body = args.get("body", "?")

        console.print()
        console.print(
            Panel(
                f"[bold red]TO:[/] {to}\n"
                f"[bold red]SUBJECT:[/] {subject}\n\n"
                f"{body}",
                border_style="red",
                title="[bold red]APPROVAL REQUIRED: send_email[/]",
            )
        )

        auto = os.getenv("AUTO_APPROVE_EMAIL", "").strip().lower()
        if auto in ("y", "yes", "1"):
            answer = "y"
        elif auto in ("n", "no", "0"):
            answer = "n"
        else:
            try:
                answer = console.input(
                    "[bold yellow]Approve this action? (y/n): [/]"
                )
            except (EOFError, KeyboardInterrupt):
                answer = "n"

        t0 = time.time()
        if answer.strip().lower() == "y":
            result = do_send_email(to=to, subject=subject, body=body)
            console.print(f"[green]Action approved: {result}[/]")
        else:
            console.print("[red]Action rejected by user.[/]")
            result = (
                "The user rejected the send_email action. "
                "Do not attempt to send it again unless they explicitly ask you to."
            )
        duration = time.time() - t0

        if tracer:
            tracer.log_tool_call(name, args, result, duration)
        return result

    # -- web_search: wrapped with retries ------------------------------
    if name == "web_search":
        query = args["query"]
        console.print(f"  [bold yellow]Search query:[/] {query}")
        t0 = time.time()
        result = retry_with_backoff(lambda: web_search(query), tracer=tracer)
        duration = time.time() - t0

        preview = result[:600]
        if len(result) > 600:
            preview += " [...]"
        console.print(
            Panel(preview, border_style="yellow", title="Search results excerpt")
        )

        if tracer:
            tracer.log_tool_call(name, args, result, duration)
        return result

    # -- recall_memory: local file, no retry needed -------------------
    elif name == "recall_memory":
        query = args["query"]
        console.print(f"  [bold yellow]Memory recall:[/] {query}")
        t0 = time.time()
        result = recall_memory(query)
        duration = time.time() - t0
        console.print(
            Panel(result[:600], border_style="magenta", title="Memory recall")
        )

        if tracer:
            tracer.log_tool_call(name, args, result, duration)
        return result

    else:
        return f"Error: unknown tool '{name}'"


# -- Agent loop ---------------------------------------------------------


def run_agent(user_goal: str, tracer: Tracer | None = None) -> str:
    """Run the research agent: plan -> think -> search -> synthesize -> critique.

    Returns the final answer string.
    """
    close_tracer = False
    if tracer is None:
        tracer = Tracer(user_goal)
        close_tracer = True

    console.print(
        Panel(f"[bold cyan]Goal:[/] {user_goal}", border_style="cyan")
    )
    console.print()

    # -- Step 1: Create the plan ---------------------------------------
    console.print("[dim]Creating research plan ...[/]")
    plan = create_plan(user_goal, tracer=tracer)
    print_plan(plan)
    tracer.log_plan(plan)

    plan_lines = "\n".join(
        f"Step {s['step']}: {s['action']}" for s in plan
    )

    # -- Step 2: Execute the plan ---------------------------------------
    messages: list[dict] = [
        {
            "role": "system",
            "content": (
                "You are a lead research and outreach assistant with web search and memory recall access. "
                "You can also send outreach emails when asked. Follow this plan:\n"
                f"{plan_lines}\n\n"
                "Use web_search to research companies, decision-makers, and market fit. "
                "Use recall_memory when the user references a lead you may have researched before. "
                "Use send_email ONLY when the user explicitly asks you to send an outreach email. "
                "When evaluating a lead, produce a short qualification summary covering: "
                "company overview, potential fit, recommended outreach angle, and next steps. "
                "Synthesise results into a clear answer and stop -- "
                "do not call a tool twice with the same query."
            ),
        },
        {"role": "user", "content": user_goal},
    ]

    answer = ""
    critique_rounds = 0
    MAX_CRITIQUE_ROUNDS = 2

    while True:
        console.print("[dim]--- Thinking ...[/]")

        t0 = time.time()
        response = retry_with_backoff(
            lambda: client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
            ),
            tracer=tracer,
        )
        latency = time.time() - t0

        if isinstance(response, str):
            console.print(f"[red]{response}[/]")
            answer = (
                "I encountered persistent errors while contacting the "
                "language model. Please try again later."
            )
            tracer.log_error(f"Main LLM loop failed: {response}")
            break

        usage = getattr(response, "usage", None)
        tracer.log_llm_call(
            model=MODEL,
            tokens_input=usage.prompt_tokens if usage else None,
            tokens_output=usage.completion_tokens if usage else None,
            latency_sec=latency,
            purpose="main_loop",
        )

        choice = response.choices[0]
        msg = choice.message

        # -- No tool call -> produce answer, then critique it ----------
        if not msg.tool_calls:
            answer = msg.content or ""
            console.print()
            console.print(
                Panel(
                    Markdown(answer),
                    border_style="green",
                    title="[bold green]Answer[/]",
                )
            )
            tracer.log_answer(answer)

            # -- Step 3: Self-critique ---------------------------------
            console.print("\n[dim]Critiquing answer ...[/]")
            verdict = critique_answer(user_goal, answer, tracer=tracer)
            print_critique(verdict)
            tracer.log_critique(verdict)

            if verdict.get("pass"):
                console.print("[dim]Answer passed critique.[/]\n")
                break
            else:
                critique_rounds += 1
                if critique_rounds >= MAX_CRITIQUE_ROUNDS:
                    console.print(
                        f"[yellow]Max critique rounds ({MAX_CRITIQUE_ROUNDS}) "
                        "reached. Accepting current answer.[/]"
                    )
                    break

                fix = verdict.get("fix", "Please improve your answer.")
                console.print(
                    f"[yellow]Critique round {critique_rounds}/"
                    f"{MAX_CRITIQUE_ROUNDS} -- asking model to improve ...[/]\n"
                )
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Please improve your answer based on this feedback: {fix}"
                        ),
                    }
                )
                continue

        # -- Tool call(s) -> execute each, feed back, loop -------------
        console.print(
            f"[bold yellow]Tool call:[/] {msg.tool_calls[0].function.name}"
        )

        messages.append(msg)

        for tool_call in msg.tool_calls:
            result = _execute_tool(tool_call, tracer=tracer)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                }
            )

        console.print("[dim]--- Looping ...[/]\n")

    # -- Step 4: Save to long-term memory ------------------------------
    save_memory(user_goal, answer)
    console.print("[dim]Saved to long-term memory.[/]")

    if close_tracer:
        log_path = tracer.path
        tracer.close()
        console.print(f"\n[dim]Trace saved to {log_path}[/]")

    return answer.strip()


if __name__ == "__main__":
    run_agent(
        "Research the company Anthropic. Determine if they're a "
        "good fit for our enterprise AI consulting service, and "
        "draft a qualification summary."
    )
