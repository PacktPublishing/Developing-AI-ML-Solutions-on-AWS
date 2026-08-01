# /// script
# dependencies = ["boto3"]
# ///
"""The guardrail seam: a real Bedrock Guardrail on AWS, a small shim locally.

A grounded assistant still needs limits: it must not leak personal data, it must
stay on topic, and its answer should be supported by the retrieved passages. On
AWS those checks are an Amazon Bedrock Guardrail applied during the model call.
Bedrock Guardrails has no local emulator, so for the offline path this module
also carries a light shim that redacts obvious personal data and refuses the
same off-topic requests. The shim is not a substitute for the managed service,
it just keeps the local loop honest about the same rules.

Usage (from the chapter root):
  make guardrail    # create the guardrail on your account
  PYTHONPATH=. uv run underwriting-agent/guardrails.py show
  PYTHONPATH=. uv run underwriting-agent/guardrails.py delete
"""

import argparse
import json
import os
import pathlib
import re

import boto3

from models import BEDROCK_REGION, TEXT_MODEL, generate

# -------------------------------------------------------------------------------
# Configuration and shim patterns
# -------------------------------------------------------------------------------
GUARDRAIL_FILE = pathlib.Path("data/guardrail.json")

BLOCKED_INPUT = "This request is outside what the underwriting assistant can help with."
BLOCKED_OUTPUT = "The assistant cannot provide that answer under the credit guardrail."

# Redaction patterns for the local shim. The managed guardrail recognizes many
# more entity types; these cover the ones that show up in deal correspondence.
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE = re.compile(r"\+?\d[\d ()\-]{7,}\d")

# Requests the assistant refuses. The managed guardrail expresses this as a
# denied topic; the shim matches the same intent with a few phrases.
DENIED_MARKERS = ["invest my", "my own money", "personal advice", "should i personally"]


# -------------------------------------------------------------------------------
# Guardrail lifecycle commands
# -------------------------------------------------------------------------------
def create() -> None:
    """Create the guardrail on Bedrock and save its id and version."""
    bedrock = boto3.client("bedrock", region_name=BEDROCK_REGION)
    guardrail = bedrock.create_guardrail(
        name="underwriting-assistant",
        description="PII redaction, off-topic refusal, and grounding for the underwriting assistant.",
        blockedInputMessaging=BLOCKED_INPUT,
        blockedOutputsMessaging=BLOCKED_OUTPUT,
        contentPolicyConfig={
            "filtersConfig": [
                {
                    "type": "PROMPT_ATTACK",
                    "inputStrength": "HIGH",
                    "outputStrength": "NONE",
                },
                {
                    "type": "MISCONDUCT",
                    "inputStrength": "MEDIUM",
                    "outputStrength": "MEDIUM",
                },
            ]
        },
        topicPolicyConfig={
            "topicsConfig": [
                {
                    "name": "PersonalFinancialAdvice",
                    "definition": (
                        "Advice on whether an individual should personally invest in,"
                        " lend to, or take a financial position in a borrower. The"
                        " assistant supports the credit process, it does not give"
                        " personal financial advice."
                    ),
                    "examples": [
                        "Should I invest my own savings in this company?",
                        "Is this borrower a good personal investment for me?",
                    ],
                    "type": "DENY",
                }
            ]
        },
        sensitiveInformationPolicyConfig={
            "piiEntitiesConfig": [
                {"type": "EMAIL", "action": "ANONYMIZE"},
                {"type": "PHONE", "action": "ANONYMIZE"},
                {"type": "CREDIT_DEBIT_CARD_NUMBER", "action": "BLOCK"},
            ]
        },
        contextualGroundingPolicyConfig={
            "filtersConfig": [
                {"type": "GROUNDING", "threshold": 0.7},
                {"type": "RELEVANCE", "threshold": 0.5},
            ]
        },
    )
    version = bedrock.create_guardrail_version(
        guardrailIdentifier=guardrail["guardrailId"], description="v1"
    )
    GUARDRAIL_FILE.parent.mkdir(exist_ok=True)
    record = {"id": guardrail["guardrailId"], "version": version["version"]}
    GUARDRAIL_FILE.write_text(json.dumps(record, indent=2))
    print(f"Created guardrail {record['id']} version {record['version']}")


def show() -> None:
    """Print the saved guardrail id and version."""
    if GUARDRAIL_FILE.exists():
        print(GUARDRAIL_FILE.read_text())
    else:
        print("No guardrail saved. Run: uv run guardrails.py create")


def delete() -> None:
    """Delete the saved guardrail from Bedrock and remove the local record."""
    if not GUARDRAIL_FILE.exists():
        print("No guardrail saved.")
        return
    record = json.loads(GUARDRAIL_FILE.read_text())
    boto3.client("bedrock", region_name=BEDROCK_REGION).delete_guardrail(
        guardrailIdentifier=record["id"]
    )
    GUARDRAIL_FILE.unlink()
    print(f"Deleted guardrail {record['id']}")


def load() -> tuple[str, str] | None:
    """Return (id, version) for the saved guardrail, or None if there is none."""
    if not GUARDRAIL_FILE.exists():
        return None
    record = json.loads(GUARDRAIL_FILE.read_text())
    return record["id"], record["version"]


# -------------------------------------------------------------------------------
# Local shim checks
# -------------------------------------------------------------------------------
def local_input_blocked(question: str) -> bool:
    """Refuse the same off-topic requests the managed guardrail would."""
    lowered = question.lower()
    return any(marker in lowered for marker in DENIED_MARKERS)


def local_redact(text: str) -> str:
    """Redact obvious personal data, the shim's output check."""
    text = EMAIL.sub("{EMAIL}", text)
    text = PHONE.sub("{PHONE}", text)
    return text


# -------------------------------------------------------------------------------
# Guarded generation
# -------------------------------------------------------------------------------
def guarded_generate(
    runtime, system: str, question: str, context: str, guardrail
) -> str:
    """Generate a grounded answer with the guardrail applied.

    On AWS the guardrail runs inside the Converse call, with the retrieved
    passages passed as the grounding source and the question as the query, so
    the contextual grounding filter can score the answer. Locally the shim runs
    the same checks in Python around a plain generation.
    """
    if os.environ.get("BEDROCK_LOCAL") == "1" or guardrail is None:
        if local_input_blocked(question):
            return BLOCKED_INPUT
        user = f"Question: {question}\n\nContext passages:\n{context}"
        return local_redact(generate(runtime, system, user))

    guardrail_id, guardrail_version = guardrail
    content = [
        {
            "guardContent": {
                "text": {"text": context, "qualifiers": ["grounding_source"]}
            }
        },
        {"guardContent": {"text": {"text": question, "qualifiers": ["query"]}}},
    ]
    resp = runtime.converse(
        modelId=TEXT_MODEL,
        system=[{"text": system}],
        messages=[{"role": "user", "content": content}],
        inferenceConfig={"maxTokens": 600, "temperature": 0.0},
        guardrailConfig={
            "guardrailIdentifier": guardrail_id,
            "guardrailVersion": guardrail_version,
        },
    )
    blocks = resp["output"]["message"]["content"]
    return "".join(b.get("text", "") for b in blocks) if blocks else BLOCKED_OUTPUT


# -------------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------------
def main() -> None:
    """Run the create, show, or delete command."""
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["create", "show", "delete"])
    args = parser.parse_args()
    {"create": create, "show": show, "delete": delete}[args.command]()


if __name__ == "__main__":
    main()
