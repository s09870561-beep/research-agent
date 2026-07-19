"""Streamlit web app wrapping the research-agent pipeline.

Usage:
    streamlit run streamlit_app.py
"""

import json
import os
import time

import streamlit as st

# ---------------------------------------------------------------------------
# API keys: st.secrets first, then os.environ
# ---------------------------------------------------------------------------
for _key in ("OPENCODE_ZEN_API_KEY", "TAVILY_API_KEY"):
    try:
        if _key in st.secrets:
            os.environ[_key] = st.secrets[_key]
    except Exception:
        pass

from dotenv import load_dotenv

load_dotenv(override=True)

from openai import OpenAI

from agent import create_plan, critique_answer, TOOLS, MODEL
from tools.search import web_search
from tools.actions import send_email as do_send_email
from memory.store import save_memory, recall_memory
from utils.retry import retry_with_backoff
from utils.tracer import Tracer

_client = OpenAI(
    base_url="https://opencode.ai/zen/v1",
    api_key=os.getenv("OPENCODE_ZEN_API_KEY", "").strip(),
)

# ---------------------------------------------------------------------------
# Custom CSS  —  Inter font, cards, buttons, spacing
# ---------------------------------------------------------------------------

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* ── Header area ──────────────────────────────────────────── */
.agent-header {
    padding: 2rem 0 0 0;
}
.agent-header h1 {
    font-size: 2.25rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: #F1F5F9;
    margin: 0 0 0.25rem 0;
}
.agent-header .accent {
    color: #2DD4BF;
}
.agent-header .tagline {
    font-size: 1rem;
    font-weight: 400;
    color: #8899AA;
    margin: 0 0 1.5rem 0;
}

/* ── Cards ──────────────────────────────────────────────── */
.result-card {
    background: #1A1E23;
    border: 1px solid #2A2F37;
    border-radius: 12px;
    padding: 1.75rem 2rem;
    margin: 1rem 0 1.5rem 0;
    box-shadow: 0 4px 20px rgba(0,0,0,0.25);
}
.result-card h3 {
    font-size: 1.1rem;
    font-weight: 600;
    color: #2DD4BF;
    margin: 0 0 0.75rem 0;
}
.result-card p, .result-card div, .result-card li {
    font-size: 0.95rem;
    line-height: 1.7;
    color: #D1D8E0;
}

/* ── Status labels ────────────────────────────────────────── */
.step-label {
    font-size: 0.85rem;
    font-weight: 500;
    color: #8899AA;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 0.15rem;
}

/* ── Buttons ─────────────────────────────────���────────────── */
.stButton > button {
    border-radius: 8px;
    font-weight: 500;
    transition: all 0.15s ease;
}
.stButton > button:active {
    transform: scale(0.97);
}

/* ── Expanders ────────────────────────────────────────────── */
.streamlit-expanderHeader {
    font-weight: 600;
    font-size: 0.9rem;
    color: #D1D8E0;
}

/* ── Info/warning boxes ────────────────────────────────────── */
.stAlert {
    border-radius: 8px;
    border-left-width: 4px;
}
"""

# ---------------------------------------------------------------------------
# Session-state
# ---------------------------------------------------------------------------

_INITIAL = {
    "phase": "input",
    "goal": "",
    "plan": None,
    "messages": None,
    "plan_steps_shown": False,
    "answer": None,
    "verdict": None,
    "critique_round": 0,
    "trace_events": [],
    "email_pending": None,
    "email_decision": None,
    "run_complete": False,
    "error": None,
    "tracer": None,
    "started_at": None,
}

for _k, _v in _INITIAL.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


def _reset():
    for _k, _v in _INITIAL.items():
        st.session_state[_k] = _v


def _add_trace(phase: str, label: str, data, duration: float | None = None):
    st.session_state.trace_events.append({
        "phase": phase, "label": label, "data": data,
        "duration": duration, "time": time.strftime("%H:%M:%S"),
    })


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Research Agent", page_icon="🔍", layout="wide")
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
st.markdown('<div class="agent-header"><h1><span class="accent">Research</span> Agent</h1><p class="tagline">AI-powered lead research and outreach — research companies, evaluate fit, and draft outreach emails.</p></div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### Controls")

    auto_approve = st.checkbox(
        "Auto-approve emails", value=False, key="auto_approve",
        help="When checked, emails are sent without asking for approval.",
    )

    if st.button("New research", use_container_width=True, type="primary"):
        _reset()
        st.rerun()

    st.divider()
    st.markdown("### API Keys")
    if os.getenv("OPENCODE_ZEN_API_KEY") and os.getenv("OPENCODE_ZEN_API_KEY") != "sk-your-key-here":
        st.success("OpenCode Zen key found")
    else:
        st.error("OpenCode Zen key missing")
    if os.getenv("TAVILY_API_KEY") and os.getenv("TAVILY_API_KEY") != "tvly-your-key-here":
        st.success("Tavily key found")
    else:
        st.error("Tavily key missing")


# ===========================================================================
# Phase machine  (unchanged logic, visual-only changes)
# ===========================================================================


def _phase_input():
    goal = st.text_area(
        "What would you like me to research?",
        height=100,
        placeholder="e.g. Research the company Anthropic and determine if they're a good fit for our enterprise AI consulting service.",
    )

    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        run_clicked = st.button("Run", type="primary", use_container_width=True)
    with col2:
        if st.button("Example", use_container_width=True):
            goal = "Research the company Stripe and evaluate whether they would benefit from our payment integration consulting services."
            st.rerun()

    if run_clicked and goal.strip():
        st.session_state.goal = goal.strip()
        st.session_state.phase = "planning"
        st.session_state.started_at = time.time()
        st.session_state.tracer = Tracer(st.session_state.goal)
        st.session_state.trace_events = []
        st.rerun()


def _phase_planning():
    status = st.status("Planning", expanded=True)
    with status:
        st.markdown('<p class="step-label">Step 1 of 4</p>', unsafe_allow_html=True)
        plan = create_plan(st.session_state.goal, tracer=st.session_state.tracer)
        st.session_state.plan = plan
        st.session_state.tracer.log_plan(plan)
        _add_trace("plan", "Research Plan", plan)
        for s in plan:
            st.write(f"**Step {s['step']}:** {s['action']}")
        st.success(f"Plan created with {len(plan)} steps")
    status.update(state="complete", label="Plan created")

    plan_lines = "\n".join(f"Step {s['step']}: {s['action']}" for s in plan)
    st.session_state.messages = [
        {
            "role": "system",
            "content": (
                "You are a lead research and outreach assistant with web search and memory recall access. "
                f"Follow this plan:\n{plan_lines}\n\n"
                "Use web_search to research companies, decision-makers, and market fit. "
                "Use recall_memory when the user references a lead you may have researched before. "
                "Use send_email ONLY when the user explicitly asks. "
                "When evaluating a lead, produce a short qualification summary covering: "
                "company overview, potential fit, recommended outreach angle, and next steps. "
                "Synthesise results into a clear answer and stop."
            ),
        },
        {"role": "user", "content": st.session_state.goal},
    ]
    st.session_state.phase = "executing"
    st.session_state.plan_steps_shown = True
    st.rerun()


def _execute_tool_streamlit(tool_call, status) -> str:
    fn = tool_call.function
    name = fn.name
    args = json.loads(fn.arguments)

    if name == "send_email":
        to = args.get("to", "?")
        subject = args.get("subject", "?")
        body = args.get("body", "?")

        if st.session_state.get("auto_approve", False):
            status.write(f"Sending email to {to} (auto-approved)")
            t0 = time.time()
            result = do_send_email(to=to, subject=subject, body=body)
            dur = time.time() - t0
            _add_trace("tool", "send_email (auto-approved)", result, dur)
            if st.session_state.tracer:
                st.session_state.tracer.log_tool_call("send_email", args, result, dur)
            return result

        st.session_state.email_pending = {
            "to": to, "subject": subject, "body": body,
            "tool_call_id": tool_call.id,
        }
        st.session_state.phase = "email_approval"
        status.update(label="Email approval required", state="running", expanded=True)
        return "__PENDING_EMAIL__"

    if name == "web_search":
        query = args["query"]
        status.write(f"Searching: {query}")
        t0 = time.time()
        result = retry_with_backoff(lambda: web_search(query), tracer=st.session_state.tracer)
        dur = time.time() - t0
        _add_trace("tool", f"web_search: {query[:60]}", result[:300], dur)
        if st.session_state.tracer:
            st.session_state.tracer.log_tool_call(name, args, result, dur)
        return result

    if name == "recall_memory":
        query = args["query"]
        status.write(f"Recalling: {query}")
        t0 = time.time()
        result = recall_memory(query)
        dur = time.time() - t0
        _add_trace("tool", f"recall_memory: {query[:60]}", result[:300], dur)
        if st.session_state.tracer:
            st.session_state.tracer.log_tool_call(name, args, result, dur)
        return result

    return f"Error: unknown tool '{name}'"


def _phase_executing():
    status = st.status("Thinking", expanded=True)

    if st.session_state.email_pending is not None:
        status.update(state="error", label="Waiting for email approval")
        return

    with status:
        st.markdown('<p class="step-label">Step 2 of 4</p>', unsafe_allow_html=True)
        t0 = time.time()
        response = retry_with_backoff(
            lambda: _client.chat.completions.create(
                model=MODEL, messages=st.session_state.messages, tools=TOOLS,
            ),
            tracer=st.session_state.tracer,
        )
        latency = time.time() - t0

        if isinstance(response, str):
            st.error(f"LLM call failed: {response}")
            st.session_state.error = response
            st.session_state.phase = "done"
            st.rerun()
            return

        usage = getattr(response, "usage", None)
        if st.session_state.tracer and usage:
            st.session_state.tracer.log_llm_call(
                model=MODEL, tokens_input=usage.prompt_tokens,
                tokens_output=usage.completion_tokens, latency_sec=latency,
                purpose="main_loop",
            )

        msg = response.choices[0].message

        if not msg.tool_calls:
            answer = msg.content or ""
            st.session_state.answer = answer
            st.session_state.tracer.log_answer(answer)
            _add_trace("answer", "Final answer", answer[:500], latency)
            st.markdown(answer)
            status.update(state="complete", label="Answer produced")
            st.session_state.phase = "critique"
            st.rerun()
            return

        status.write(f"**Tool:** {msg.tool_calls[0].function.name}")
        st.session_state.messages.append(msg)

        for tool_call in msg.tool_calls:
            result = _execute_tool_streamlit(tool_call, status)
            if result == "__PENDING_EMAIL__":
                return
            st.session_state.messages.append({
                "role": "tool", "tool_call_id": tool_call.id, "content": result,
            })
            with st.expander(f"Result preview", expanded=False):
                st.text(result[:1500])

        status.write("Looping ...")

    status.update(state="complete", label=f"Executed: {msg.tool_calls[0].function.name}")
    st.rerun()


def _phase_email_approval():
    pending = st.session_state.email_pending
    if not pending:
        st.session_state.phase = "executing"
        st.rerun()
        return

    st.warning("**Email approval required**")
    st.markdown(f"**To:** {pending['to']}")
    st.markdown(f"**Subject:** {pending['subject']}")
    st.markdown("**Body:**")
    st.code(pending['body'], language="text")

    tid = pending.get("tool_call_id", "__email__")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Approve", type="primary", use_container_width=True):
            t0 = time.time()
            result = do_send_email(to=pending["to"], subject=pending["subject"], body=pending["body"])
            dur = time.time() - t0
            st.session_state.messages.append({
                "role": "tool", "tool_call_id": tid, "content": result,
            })
            _add_trace("tool", "send_email (approved)", result, dur)
            if st.session_state.tracer:
                st.session_state.tracer.log_tool_call("send_email", pending, result, dur)
            st.success(result)
            st.session_state.email_pending = None
            st.session_state.phase = "executing"
            st.rerun()

    with col2:
        if st.button("Reject", use_container_width=True):
            result = ("The user rejected the send_email action. "
                      "Do not attempt to send it again unless they explicitly ask to.")
            st.session_state.messages.append({
                "role": "tool", "tool_call_id": tid, "content": result,
            })
            _add_trace("tool", "send_email (rejected)", result)
            if st.session_state.tracer:
                st.session_state.tracer.log_tool_call("send_email", pending, result, 0)
            st.info("Email was rejected.")
            st.session_state.email_pending = None
            st.session_state.phase = "executing"
            st.rerun()


def _phase_critique():
    status = st.status("Critiquing answer", expanded=True)

    with status:
        st.markdown('<p class="step-label">Step 3 of 4</p>', unsafe_allow_html=True)
        verdict = critique_answer(
            st.session_state.goal, st.session_state.answer,
            tracer=st.session_state.tracer,
        )
        st.session_state.verdict = verdict
        st.session_state.tracer.log_critique(verdict)

        passed = verdict.get("pass", False)
        reason = verdict.get("reason", "")
        fix = verdict.get("fix", "")

        if passed:
            st.success(f"Critique passed: {reason}")
            _add_trace("critique", "Critique passed", {"reason": reason})
            status.update(state="complete", label="Critique passed")
            st.session_state.phase = "done"
        else:
            st.warning(f"Critique failed: {reason}")
            _add_trace("critique", "Critique failed", {"reason": reason, "fix": fix})
            status.update(state="error", label="Critique failed")

            st.session_state.critique_round += 1
            max_rounds = 2
            if st.session_state.critique_round >= max_rounds:
                st.info(f"Max critique rounds ({max_rounds}) reached. Accepting current answer.")
                status.update(state="complete", label=f"Reached max critique rounds ({max_rounds})")
                st.session_state.phase = "done"
            else:
                improvement = fix or "Please improve your answer."
                st.write(f"Asking model to improve (round {st.session_state.critique_round}/{max_rounds}) ...")
                st.session_state.messages.append({
                    "role": "user",
                    "content": f"Please improve your answer based on this feedback: {improvement}",
                })
                status.update(state="running", label=f"Improvement round {st.session_state.critique_round}")
                st.session_state.phase = "executing"

    st.rerun()


def _phase_done():
    answer = st.session_state.answer
    verdict = st.session_state.verdict
    plan = st.session_state.plan

    st.markdown("## Result")

    # Card: Final answer
    st.markdown('<div class="result-card"><h3>Final Answer</h3>', unsafe_allow_html=True)
    st.markdown(answer)
    st.markdown('</div>', unsafe_allow_html=True)

    # Card: Critique
    if verdict:
        passed = verdict.get("pass", False)
        reason = verdict.get("reason", "")
        fix = verdict.get("fix", "")
        status_icon = "passed" if passed else "failed"
        st.markdown(f'<div class="result-card"><h3>Critique — {status_icon}</h3>', unsafe_allow_html=True)
        st.markdown(reason)
        if fix:
            st.markdown(f"**Suggested improvement:** {fix}")
        st.markdown('</div>', unsafe_allow_html=True)

    save_memory(st.session_state.goal, answer)
    st.caption("Saved to long-term memory.")

    # Trace
    with st.expander("Show reasoning trace", expanded=False):
        st.markdown("### Research Plan")
        if plan:
            for s in plan:
                st.markdown(f"**Step {s['step']}:** {s['action']}")
        st.markdown("### Events")
        for evt in st.session_state.trace_events:
            dur_str = f" ({evt['duration']:.1f}s)" if evt.get("duration") else ""
            st.markdown(f"**[{evt['time']}]** {evt['label']}{dur_str}")
        st.markdown("### Critique Verdict")
        if verdict:
            st.json(verdict)

    if st.session_state.tracer:
        st.caption(f"Trace saved to `{st.session_state.tracer.path}`")

    st.divider()
    if st.button("Research something else", type="primary"):
        _reset()
        st.rerun()


# ===========================================================================
# Phase router
# ===========================================================================

phase = st.session_state.phase

if phase != "input" and st.session_state.goal:
    st.info(f"**Researching:** {st.session_state.goal}")

if phase == "input":
    _phase_input()
elif phase == "planning":
    _phase_planning()
elif phase == "executing":
    _phase_executing()
elif phase == "email_approval":
    _phase_email_approval()
elif phase == "critique":
    _phase_critique()
elif phase == "done":
    _phase_done()
