# Lead Research & Outreach Agent

An open-source CRM lead qualification assistant that uses an LLM (via OpenCode Zen) with
web search, long-term memory, an email approval gate, self-critique, and
structured observability logging to research companies, qualify leads, and
draft personalized outreach emails.

Built as a portfolio project to demonstrate agentic patterns in Python.

---

## Architecture

The agent runs in a staged loop — it doesn't just fire off one LLM call
and hope for the best.

1.  **Plan** — Given a user goal, the LLM first breaks it into 2–5
    concrete research steps (JSON output).
2.  **Tool-calling loop** — The agent follows the plan, calling
    `web_search` (Tavily) or `recall_memory` (past Q&A) as needed, and
    feeds tool results back into the conversation until it has enough
    information.
3.  **Email approval gate** — If the agent attempts to send an email,
    the terminal pauses and shows exactly what would be sent. The user
    must type `y` to approve; otherwise the action is rejected.
4.  **Self-critique** — After producing a final answer, the agent calls
    the LLM again as a strict reviewer. If the answer fails, the fix
    is fed back as a new user message (up to 2 revision rounds).
5.  **Memory** — Every Q&A pair is saved to a local JSON file. On
    future runs the agent can call `recall_memory` to find relevant
    past research by keyword matching.
6.  **Tracing** — All events (plan, LLM calls with token counts, tool
    calls with latency, critique verdicts, errors, retries) are written
    to a JSONL file in `logs/` for later replay.

---

## Setup

```bash
# 1. Clone (or cd into the project)
cd research-agent

# 2. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install openai tavily-python python-dotenv rich

# 4. Copy the env template and fill in your keys
cp .env.example .env
```

You'll need two free API keys:

| Service | Get one at | Used for |
|---------|------------|----------|
| [OpenCode Zen](https://opencode.ai/auth) | https://opencode.ai/auth | LLM calls (free account with billing details required) |
| [Tavily](https://tavily.com/) | https://tavily.com/ | Web search |

---

## How to run

```bash
python3 agent.py
```

The agent prints a plan, then thinks / searches / remembers in a loop,
and finishes with a final answer and a self-critique.

**Questions that work well:**
- *Research the company Anthropic and determine if they're a good
  fit for our enterprise AI consulting service.*
- *Research Stripe and draft an outreach email about payment
  integration consulting to their partnerships team.*

Set `AUTO_APPROVE_EMAIL=y` in your environment to auto-approve email
actions without a terminal prompt (useful for scripting):

```bash
export AUTO_APPROVE_EMAIL=y    # Linux / macOS
$env:AUTO_APPROVE_EMAIL='y'   # Windows PowerShell
```

---

## How to view a trace

Every run logs structured events to `logs/run_<timestamp>.jsonl`.

```bash
# Auto-detect the latest trace
python3 view_trace.py

# Or specify one
python3 view_trace.py logs/run_20260718_160258.jsonl
```

The viewer renders a styled timeline with goal, plan, LLM calls (model,
tokens, latency), tool calls (args, duration, result preview), answer,
critique verdict, and any errors or retries.

---

## How to run evals

```bash
python3 evals/run_evals.py
```

This runs 8 test cases (lead qualification questions, knowledge questions, an email
test), judges each answer against a plain-English criteria
string, and prints a summary table with pass/fail counts.

**Latest results (OpenCode Zen, deepseek-v4-flash-free):**

```
Eval Results: 8/8 passed (100%)
```

The suite covers:
- **Lead qualification questions** — "Research Anthropic and qualify them"
- **Knowledge questions** — "What is the capital of France?"
- **Web-search-dependent questions** — "Research Acme Corporation"
- **Tool-action questions** — "Research Stripe and draft an outreach email"

Each answer is self-critiqued before the judge evaluates it, and
all events are traced to `logs/` for inspection.

---

## Design decisions

**Plan-first.**  Asking the model to decompose a goal before executing
reduces the chance of missing important sub-topics.  The plan is also
shown to the user, so they see the agent's strategy upfront.

**Explicit step-by-step prompting (no agents SDK).**  The agent doesn't
use LangChain, CrewAI, or any framework.  Every part of the loop is
visible in `agent.py` — tool registration, message assembly, retries,
approval gates.  This makes the system predictable and easy to debug,
and keeps the dependency list minimal.

**Approval gate for email.**  Actions that can't be undone (sending a
message, posting to an API) should always require human confirmation.
The gate prints exactly what would be sent and blocks until the user
responds.  The model is told not to retry if the user says no.

**Retry with backoff.**  Both the OpenCode Zen LLM call and the Tavily
search API can fail transiently (rate limits, network blips).  Rather
than crash, the agent retries up to 3 times with exponential backoff
(1 s, 2 s, 4 s) and returns a graceful error string if all attempts
fail.  The loop continues and the model adapts.

**Self-critique.**  Having the LLM review its own answer catches
hallucinations, omissions, and vague claims.  A 2-round cap prevents
infinite revision loops.  The critique is run with a different
temperature (0.2 vs the main loop's default) so the reviewer is
more consistent.

**JSONL tracing.**  A JSON file per run with one event per line is
trivially parseable, grep-able, and compact.  It captures token
counts, latencies, and tool durations — data that helps understand
where time and budget go.

---

## Known limitations

**OpenRouter rate limit (encountered and resolved).**  The project
originally used OpenRouter's free tier, which caps at 50 requests per
day.  During eval runs this limit was consistently hit halfway through
the suite, causing judge calls to fail and suppressing the pass rate.
The fix was switching the provider to **OpenCode Zen**
(`deepseek-v4-flash-free`) — an OpenAI-compatible endpoint that handles
the same workload without daily rate limits.  The retry-with-backoff
pattern kept the agent working throughout; no code beyond the client
config and model name needed to change.  This is a good example of
adapting to a live infrastructure constraint: the agent's
provider-agnostic design made the swap trivial.

**Memory is keyword-based.**  `recall_memory` simply checks whether
query words appear in saved question strings — no semantic search, no
embeddings.  At larger scale you'd swap the JSON file for a vector
database (Chroma, Qdrant) and use embedding-based similarity search.

**Single-threaded loop.**  The agent handles one question at a time
with no parallelism.  For batch research you'd want a queue system
and concurrent workers.

**No streaming.**  The LLM response is waited for in full before any
output appears.  Streaming would make long searches feel snappier.

**Windows Unicode.**  On Windows the terminal's default encoding
(cp1252) can't display some Unicode characters.  The code uses
`PYTHONIOENCODING=utf-8` and a `_clean()` helper, but rich-styled
output works best in a modern terminal (Windows Terminal, VS Code
terminal, or a Unix terminal).
