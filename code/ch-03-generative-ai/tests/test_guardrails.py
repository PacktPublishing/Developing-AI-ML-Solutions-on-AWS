"""The local guardrail shim: the offline stand-in for Bedrock Guardrails.

Pure unit tests. They check that the shim refuses the same off-topic requests
and redacts the same personal data the managed guardrail is configured for.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "underwriting-agent"))

from guardrails import local_input_blocked, local_redact  # noqa: E402


def test_blocks_personal_advice():
    assert (
        local_input_blocked("Should I invest my own savings in this company?") is True
    )


def test_allows_underwriting_questions():
    assert (
        local_input_blocked("What DSCR floor applies to a solar project company?")
        is False
    )


def test_redacts_email_and_phone():
    out = local_redact("Reach the sponsor at jane@acme.com or +1 415 555 0100.")
    assert "jane@acme.com" not in out
    assert "{EMAIL}" in out
    assert "{PHONE}" in out
