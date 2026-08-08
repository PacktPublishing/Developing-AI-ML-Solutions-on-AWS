"""The local guardrail shim: the offline stand-in for Bedrock Guardrails."""

from guardrails import local_input_blocked, local_redact


def test_blocks_personal_advice():
    """Personal investment advice is blocked."""
    assert (
        local_input_blocked("Should I invest my own savings in this company?") is True
    )


def test_allows_underwriting_questions():
    """An underwriting question is allowed through."""
    assert (
        local_input_blocked("What DSCR floor applies to a solar project company?")
        is False
    )


def test_redacts_email_and_phone():
    """Email and phone are redacted from the input."""
    out = local_redact("Reach the sponsor at jane@acme.com or +1 415 555 0100.")
    assert "jane@acme.com" not in out
    assert "{EMAIL}" in out
    assert "{PHONE}" in out
