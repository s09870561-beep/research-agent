"""Simulated external actions that require user approval."""


def send_email(to: str, subject: str, body: str) -> str:
    """Simulate sending an email.

    In a real app this would call an email API. Here we just log it
    so the demo can show the approval gate working.
    """
    print(f"EMAIL SENT to {to}: {subject}")
    return f"Email successfully sent to {to} with subject '{subject}'."
