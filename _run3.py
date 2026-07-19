"""Run demo.py 3 times in sequence and capture the final answer from each."""
import subprocess
import os
import sys

PYTHON = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "Scripts", "python.exe")

QUESTION = "Research the top open-source AI agent frameworks for 2026. Compare LangGraph, CrewAI, and AutoGen on features, ecosystem size, and ease of use."

for i in range(1, 4):
    print(f"\n{'='*70}")
    print(f"  RUN {i}/3")
    print(f"{'='*70}\n")
    result = subprocess.run(
        [PYTHON, "demo.py"],
        input=QUESTION + "\n",
        capture_output=True,
        text=True,
        timeout=180,
        cwd=os.path.dirname(os.path.abspath(__file__)),
        env={**os.environ, "PYTHONIOENCODING": "utf-8", "AUTO_APPROVE_EMAIL": "y"},
    )
    stdout = result.stdout + result.stderr
    # Extract what's between the final answer panel markers
    sections = stdout.split("Final Answer")
    if len(sections) >= 2:
        part = sections[1]
        if "Demo complete." in part:
            part = part[: part.index("Demo complete.")]
        # strip panel borders
        lines = [l.strip() for l in part.split("\n") if l.strip() and not l.strip().startswith("┌") and not l.strip().startswith("│") and not l.strip().startswith("└") and not l.strip().startswith("─")]
        answer_text = "\n".join(lines)
    else:
        answer_text = stdout[-1000:] if stdout else "(no output)"
    print(f"\n--- Final Answer (Run {i}) ---\n{answer_text[:800]}")
    if len(answer_text) > 800:
        print(" ... (truncated)")
    has_empty = not answer_text.strip()
    print(f"\n  -> Answer empty? {'YES - PROBLEM' if has_empty else 'NO - OK'}")
    if has_empty:
        print("  RAW OUTPUT DUMP:")
        print(stdout[-2000:])
        sys.exit(1)
