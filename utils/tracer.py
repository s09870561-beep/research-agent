"""Observability logger — writes structured event logs in JSONL format."""

import json
import os
from datetime import datetime

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")


def _run_timestamp() -> str:
    """Return a compact, sortable timestamp for the run filename."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class Tracer:
    """Structured event logger for one agent run.

    Usage:
        tracer = Tracer("What are the best AI frameworks?")
        tracer.log_plan([{"step": 1, "action": "..."}])
        tracer.log_llm_call(model="...", tokens=100, latency=2.3)
        tracer.log_tool_call("web_search", {"query": "..."}, "result ...", 1.5)
        tracer.log_answer("Final answer text")
        tracer.log_critique({"pass": True, "reason": "..."})
        tracer.log_error("Something went wrong")
    """

    def __init__(self, user_goal: str):
        os.makedirs(LOG_DIR, exist_ok=True)
        ts = _run_timestamp()
        self.path = os.path.join(LOG_DIR, f"run_{ts}.jsonl")
        self._file = open(self.path, "w", encoding="utf-8")
        # Log the goal as the first event
        self._write("goal", {"goal": user_goal})

    # ------------------------------------------------------------------
    def _write(self, event: str, data: dict) -> None:
        entry = {"timestamp": datetime.now().isoformat(timespec="milliseconds"), "event": event, **data}
        self._file.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._file.flush()

    # ------------------------------------------------------------------
    def log_plan(self, steps: list[dict]) -> None:
        self._write("plan", {"steps": steps})

    def log_llm_call(
        self,
        model: str,
        tokens_input: int | None,
        tokens_output: int | None,
        latency_sec: float,
        purpose: str = "",
    ) -> None:
        self._write(
            "llm_call",
            {
                "model": model,
                "tokens_input": tokens_input,
                "tokens_output": tokens_output,
                "latency_sec": round(latency_sec, 3),
                "purpose": purpose,
            },
        )

    def log_tool_call(
        self,
        tool: str,
        args: dict,
        result_preview: str,
        duration_sec: float,
    ) -> None:
        self._write(
            "tool_call",
            {
                "tool": tool,
                "args": args,
                "result_preview": result_preview[:500],
                "duration_sec": round(duration_sec, 3),
            },
        )

    def log_answer(self, answer: str) -> None:
        self._write("answer", {"answer": answer})

    def log_critique(self, verdict: dict) -> None:
        self._write("critique", {"verdict": verdict})

    def log_error(self, message: str) -> None:
        self._write("error", {"error": message})

    def log_retry(self, attempt: int, max_retries: int, error: str) -> None:
        self._write(
            "retry",
            {
                "attempt": attempt,
                "max_retries": max_retries,
                "error": str(error),
            },
        )

    # ------------------------------------------------------------------
    def close(self) -> str:
        self._file.close()
        return self.path

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
