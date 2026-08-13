# /// script
# dependencies = ["boto3", "ollama"]
# ///
"""Zero-shot conversation classification through Bedrock batch inference.

Turns each labeled conversation into a Converse request, ships the JSONL manifest
through a batch inference job (the local shim or real Bedrock), collects the
.jsonl.out responses, and maps each answer back to one of the known categories.

Usage (from the chapter root):
  BEDROCK_LOCAL=1 S3_ENDPOINT=http://localhost:9000 \
    uv run src/classify.py --dataset conversations --bucket ch10-batch
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

from batch import get_bedrock, s3_client
from models import TEXT_MODEL, answer_text

INSTRUCTION = (
    "You are a bank's conversation router. Classify the conversation into exactly"
    " one category from the list. Reply with only the category name, nothing else."
)

INSTRUCTION_PROBS = (
    "You are a bank's conversation router. Read the conversation and give the"
    " probability that it belongs to each category. Reply with only a JSON object"
    " mapping every category to a probability in [0,1], summing to about 1."
)

# Variant B draws the category borders instead of listing bare names: a one-line definition
# per category plus two tie-break rules aimed at the leaks variant A shows (loan requests and
# app/login issues collapsing into the broad-sounding classes). Select with --variant b.
INSTRUCTION_B = (
    "You are a bank's conversation router. Choose the ONE category the conversation is"
    " mainly about, using these definitions:\n"
    "- Bank Accounts: current/checking accounts, statements, fees, balances.\n"
    "- Cards: debit/credit cards, activation, limits, lost or stolen, card fraud.\n"
    "- Customer Rewards: points, cashback, loyalty, referrals.\n"
    "- Digital Banking: the mobile app or online banking, login, passwords, transfers, errors.\n"
    "- Insurance: policies and claims (life, home, travel, health).\n"
    "- Investment Funds: mutual funds, ETFs, brokerage, buying/selling securities.\n"
    "- Mortgages: home/property loans, rates, refinancing.\n"
    "- Pension Plans: retirement/pension contributions and withdrawals.\n"
    "- Personal Loans: unsecured consumer loans and their repayment (not property).\n"
    "- Savings & Deposits: savings accounts and fixed/term deposits and their interest ONLY.\n"
    "Rules: borrowing money is Personal Loans or Mortgages, never Savings & Deposits;"
    " app/login/transfer issues are Digital Banking."
)

# Variant C adds few-shot examples on top of B's borders: one canonical utterance per
# category, so the model sees "if it says this, it is that" rather than only definitions.
# These are written fresh (not drawn from the scored dataset) to avoid leaking test items.
INSTRUCTION_C = (
    INSTRUCTION_B + "\n\nExamples:\n"
    '"I would like to open a current account" -> Bank Accounts\n'
    '"My debit card was declined at the till" -> Cards\n'
    '"How do I redeem my cashback points?" -> Customer Rewards\n'
    '"The mobile app will not let me log in" -> Digital Banking\n'
    '"I need to make a claim on my home policy" -> Insurance\n'
    '"I want to buy units in an equity fund" -> Investment Funds\n'
    '"What is the rate on a 25-year home loan?" -> Mortgages\n'
    '"How much can I pay into my pension this year?" -> Pension Plans\n'
    '"Can I get a 5,000 loan to consolidate debt?" -> Personal Loans\n'
    '"What interest does your 12-month fixed deposit pay?" -> Savings & Deposits'
)


def model_input(
    text: str, labels: list[str], distribution: bool = False, variant: str = "a"
) -> dict:
    """Build the Converse request body: a single category, or a probability per category."""
    if distribution:
        instruction = INSTRUCTION_PROBS
    elif variant == "c":
        instruction = INSTRUCTION_C
    elif variant == "b":
        instruction = INSTRUCTION_B
    else:
        instruction = INSTRUCTION
    tail = "\n\nJSON:" if distribution else "\n\nCategory:"
    prompt = (
        f"{instruction}\n\nCategories:\n- "
        + "\n- ".join(labels)
        + f"\n\nConversation:\n{text}{tail}"
    )
    return {
        "messages": [{"role": "user", "content": [{"text": prompt}]}],
        "inferenceConfig": {"maxTokens": 300 if distribution else 32, "temperature": 0},
    }


def _parse_probs(text: str, labels: list[str]) -> dict[str, float]:
    """Parse a JSON category->probability object, tolerating code fences; normalize over labels."""
    text = re.sub(r"^```[\w-]*\s*|\s*```$", "", text.strip())
    start, end = text.find("{"), text.rfind("}")
    try:
        raw = json.loads(text[start : end + 1]) if start >= 0 else {}
    except json.JSONDecodeError:
        raw = {}
    lower = {label.lower(): label for label in labels}
    probs = {label: 0.0 for label in labels}
    for key, value in raw.items():
        label = lower.get(str(key).lower())
        if label is not None:
            probs[label] = max(0.0, float(value))
    total = sum(probs.values())
    return {k: v / total for k, v in probs.items()} if total else probs


def write_manifest(
    rows: list[dict],
    labels: list[str],
    bucket: str,
    key: str,
    s3,
    distribution: bool,
    variant: str = "a",
) -> str:
    """Write the {recordId, modelInput} JSONL manifest to S3; return its s3 URI."""
    lines = [
        json.dumps(
            {
                "recordId": r["id"],
                "modelInput": model_input(r["text"], labels, distribution, variant),
            }
        )
        for r in rows
    ]
    s3.put_object(Bucket=bucket, Key=key, Body="\n".join(lines).encode("utf-8"))
    return f"s3://{bucket}/{key.rsplit('/', 1)[0]}/"


def collect(
    bucket: str,
    out_prefix: str,
    job_arn: str,
    labels: list[str],
    s3,
    distribution: bool,
) -> dict[str, dict]:
    """Read the .jsonl.out shards; return {recordId: {pred, probs}} per record."""
    job_id = job_arn.split("/")[-1]
    prefix = (
        f"{out_prefix.removeprefix('s3://').partition('/')[2].rstrip('/')}/{job_id}/"
    )
    lower = {label.lower(): label for label in labels}
    preds: dict[str, dict] = {}
    for obj in s3.list_objects_v2(Bucket=bucket, Prefix=prefix).get("Contents", []):
        # only the record shards; Bedrock also writes a manifest.json.out summary
        if not obj["Key"].endswith(".jsonl.out"):
            continue
        body = (
            s3.get_object(Bucket=bucket, Key=obj["Key"])["Body"].read().decode("utf-8")
        )
        for line in body.splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            said = answer_text(rec["modelOutput"])
            if distribution:
                probs = _parse_probs(said, labels)
                pred = max(probs, key=lambda k: probs[k]) if any(probs.values()) else ""
                preds[rec["recordId"]] = {"pred": pred, "probs": probs}
            else:
                match = next(
                    (orig for low, orig in lower.items() if low in said.lower()), None
                )
                preds[rec["recordId"]] = {"pred": match or said.lower(), "probs": None}
    return preds


def classify(
    rows: list[dict],
    labels: list[str],
    bucket: str,
    distribution: bool = False,
    variant: str = "a",
) -> dict[str, dict]:
    """Run the whole batch flow and return {conversation id: {pred, probs}}."""
    s3 = s3_client()
    in_uri = write_manifest(
        rows, labels, bucket, "input/manifest.jsonl", s3, distribution, variant
    )
    out_uri = f"s3://{bucket}/output/"
    bedrock = get_bedrock()
    # roleArn is required on AWS (the service role Bedrock assumes to reach S3);
    # the local shim ignores it. BATCH_MODEL lets the cloud round pick a
    # batch-capable model without changing the seam's default.
    job = bedrock.create_model_invocation_job(
        jobName=f"classify-{int(time.time())}",
        roleArn=os.environ.get("BEDROCK_BATCH_ROLE", ""),
        modelId=os.environ.get("BATCH_MODEL", TEXT_MODEL),
        inputDataConfig={
            "s3InputDataConfig": {"s3Uri": in_uri, "s3InputFormat": "JSONL"}
        },
        outputDataConfig={"s3OutputDataConfig": {"s3Uri": out_uri}},
        modelInvocationType="Converse",
    )
    arn = job["jobArn"]
    while True:
        status = bedrock.get_model_invocation_job(jobIdentifier=arn)["status"]
        if status in ("Completed", "Failed", "Stopped"):
            print("job", status)
            break
        time.sleep(10)
    return collect(bucket, out_uri, arn, labels, s3, distribution)


def main() -> None:
    """Classify a prepared dataset and write predictions.jsonl."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="conversations")
    p.add_argument("--bucket", default="ch10-batch")
    p.add_argument("--data", type=Path, default=Path("data/generated"))
    p.add_argument("--limit", type=int, default=0)
    p.add_argument(
        "--probs", action="store_true", help="ask for a probability per category"
    )
    p.add_argument(
        "--variant",
        default="a",
        choices=["a", "b", "c"],
        help="prompt variant: a (bare names), b (defined borders), c (borders + few-shot)",
    )
    a = p.parse_args()

    rows = [
        json.loads(line)
        for line in (a.data / f"{a.dataset}.jsonl").read_text().splitlines()
    ]
    if a.limit:
        rows = rows[: a.limit]
    labels = json.loads((a.data / f"{a.dataset}.labels.json").read_text())

    preds = classify(rows, labels, a.bucket, a.probs, a.variant)
    out = a.data / f"{a.dataset}.predictions.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            got = preds.get(r["id"], {"pred": "", "probs": None})
            record = {"id": r["id"], "label": r["label"], "pred": got["pred"]}
            if got.get("probs"):
                record["probs"] = got["probs"]
            f.write(json.dumps(record) + "\n")
    hits = sum(preds.get(r["id"], {}).get("pred") == r["label"] for r in rows)
    print(f"{hits}/{len(rows)} correct -> {out}")


if __name__ == "__main__":
    main()
