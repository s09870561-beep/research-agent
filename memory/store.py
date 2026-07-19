"""Long-term memory store backed by a local JSON file."""

import json
import os
from datetime import datetime

MEMORY_DIR = os.path.dirname(os.path.abspath(__file__))
MEMORY_FILE = os.path.join(MEMORY_DIR, "memory.json")


def _load() -> list[dict]:
    """Load all memory entries from disk."""
    if not os.path.exists(MEMORY_FILE):
        return []
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def _save(entries: list[dict]) -> None:
    """Write all memory entries to disk."""
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)


def save_memory(question: str, answer: str) -> str:
    """Store a Q&A pair in long-term memory.

    Args:
        question: The user's research question.
        answer: The agent's final answer.

    Returns:
        A confirmation string.
    """
    entries = _load()
    entries.append(
        {
            "question": question,
            "answer": answer,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
    )
    _save(entries)
    return f"Saved to memory ({len(entries)} entries total)."


def recall_memory(query: str) -> str:
    """Search past Q&A entries by keyword match on the question field.

    Args:
        query: Search words to match against saved questions.

    Returns:
        A formatted string of matching past entries, or a "not found" message.
    """
    entries = _load()
    if not entries:
        return "No relevant past research found."

    query_lower = query.lower()
    query_words = query_lower.split()

    matches = []
    for entry in entries:
        q = entry.get("question", "").lower()
        # Score: count how many query words appear in the question
        score = sum(1 for word in query_words if word in q)
        if score > 0:
            matches.append((score, entry))

    if not matches:
        return "No relevant past research found."

    # Sort by score descending, take top 3
    matches.sort(key=lambda x: x[0], reverse=True)
    top = matches[:3]

    parts = ["Past research found in memory:\n"]
    for score, entry in top:
        ts = entry.get("timestamp", "unknown date")
        parts.append(f"[{ts}] {entry.get('question', '?')}")
        parts.append(f"  Answer: {entry.get('answer', '')[:300]}")
        parts.append("")

    return "\n".join(parts).strip()
