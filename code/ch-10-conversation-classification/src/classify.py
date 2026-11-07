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
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import categories
from batch import get_bedrock, s3_client
from runs import new_run_id
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

# Variant B defines each category and adds rules for overlapping topics.
# The definitions live in categories.py because the decision model in decide.py is
# briefed from the same text, so the two routers can be compared fairly.
INSTRUCTION_B = (
    "You are a bank's conversation router. Choose the ONE category the conversation is"
    " mainly about, using these definitions:\n"
    + categories.as_lines()
    + "\n"
    + categories.RULES
)

# Variant C adds one example per category, written independently of the dataset.
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


# Stop polling when the job reaches any of these statuses.
TERMINAL = ("Completed", "PartiallyCompleted", "Failed", "Stopped", "Expired")
# Job statuses eligible for output collection.
USABLE = ("Completed", "PartiallyCompleted")


@dataclass
class Run:
    """Batch job status, predictions, record errors and service counts."""

    status: str
    message: str
    # recordId -> {"pred": str, "probs": dict | None}
    predictions: dict[str, dict] = field(default_factory=dict)
    # recordId -> error reason
    failures: dict[str, str] = field(default_factory=dict)
    # Record counts returned by the service, when available.
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        """Return whether the job status permits saving its results."""
        return self.status in USABLE


def _norm(text: str) -> str:
    """Reduce a category name or an answer to comparable words."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower().replace("&", " and ")).strip()


def _loose(text: str) -> str:
    """Normalise further, so "Savings and Deposits" matches "Savings & Deposits"."""
    return _norm(text).replace(" and ", " ")


def _clean(said: str) -> str:
    """Remove surrounding formatting from a category answer."""
    text = re.sub(r"^```[\w-]*\s*|\s*```$", "", said.strip())
    text = re.sub(r"(?i)^(category|answer|label)\s*:\s*", "", text.strip())
    return text.strip().strip("\"'*").strip(" .\t\n")


def parse_label(said: str, labels: list[str]) -> tuple[str, str]:
    """Return a recognised category or an error reason.

    Accept an exact match after normalisation, or text containing exactly one
    recognised category. Reject answers containing multiple categories.
    """
    cleaned = _clean(said)
    if not cleaned:
        return "", "the model answered with nothing"
    exact = {_loose(label): label for label in labels}
    if cleaned_label := exact.get(_loose(cleaned)):
        return cleaned_label, ""
    words = _loose(cleaned)
    mentioned = [
        label
        for label in labels
        if re.search(rf"\b{re.escape(_loose(label))}\b", words)
    ]
    if len(mentioned) == 1:
        return mentioned[0], ""
    if len(mentioned) > 1:
        return "", f"names {len(mentioned)} categories: {said.strip()[:80]}"
    return "", f"names no category: {said.strip()[:80]}"


def parse_probs(said: str, labels: list[str]) -> tuple[dict[str, float], str]:
    """Parse and validate a probability distribution over the supplied labels.

    Return the probabilities and an error reason when validation fails.
    """
    text = re.sub(r"^```[\w-]*\s*|\s*```$", "", said.strip())
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        return {}, f"no JSON object in the answer: {said.strip()[:80]}"
    try:
        raw = json.loads(text[start : end + 1])
    except json.JSONDecodeError as err:
        return {}, f"the JSON object does not parse: {err}"
    if not isinstance(raw, dict):
        return {}, "the JSON is not an object of category to probability"

    by_label = {_loose(label): label for label in labels}
    probs = dict.fromkeys(labels, 0.0)
    named = 0
    for key, value in raw.items():
        label = by_label.get(_loose(str(key)))
        if label is None:
            # Ignore keys outside the supplied category set.
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            return {}, f"{key} is not a number: {value!r}"
        try:
            number = float(value)
        except ValueError:
            return {}, f"{key} is not a number: {value!r}"
        if not math.isfinite(number):
            return {}, f"{key} is not finite: {value!r}"
        if not 0.0 <= number <= 1.0:
            return {}, f"{key} is outside [0,1]: {value!r}"
        probs[label] = number
        named += 1
    if not named:
        return {}, "the answer names none of the categories"
    total = sum(probs.values())
    # Reject probability totals outside the accepted tolerance.
    if not 0.5 <= total <= 1.5:
        return {}, f"the probabilities total {total:.3f}, not about 1"
    return {label: value / total for label, value in probs.items()}, ""


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


def _record_error(rec: dict) -> str:
    """Extract the record error, or report that both output and error are missing."""
    for key in ("error", "errorMessage", "message"):
        if rec.get(key):
            value = rec[key]
            return str(value if isinstance(value, str) else json.dumps(value))[:200]
    return "the record carries no modelOutput and no error"


def _cut_off(model_output: dict) -> bool:
    """Return whether generation stopped at the output token limit."""
    return str(model_output.get("stopReason", "")) == "max_tokens"


def collect(
    bucket: str,
    out_prefix: str,
    job_arn: str,
    labels: list[str],
    s3,
    distribution: bool,
) -> tuple[dict[str, dict], dict[str, str]]:
    """Read output shards and return predictions and per-record errors.

    Continue collecting after an inference error or an invalid model response.
    """
    job_id = job_arn.split("/")[-1]
    prefix = (
        f"{out_prefix.removeprefix('s3://').partition('/')[2].rstrip('/')}/{job_id}/"
    )
    preds: dict[str, dict] = {}
    failures: dict[str, str] = {}
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
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as err:
                failures[f"{obj['Key']}:{len(failures)}"] = f"unreadable line: {err}"
                continue
            record_id = str(rec.get("recordId", ""))
            if not record_id:
                failures[f"{obj['Key']}:{len(failures)}"] = "record without a recordId"
                continue
            if "modelOutput" not in rec:
                failures[record_id] = _record_error(rec)
                continue
            try:
                said = answer_text(rec["modelOutput"])
            except (KeyError, IndexError, TypeError) as err:
                failures[record_id] = f"the answer is not a Converse response: {err}"
                continue
            cut_off = _cut_off(rec["modelOutput"])
            if distribution:
                probs, why = parse_probs(said, labels)
                if why:
                    failures[record_id] = (
                        f"the answer stopped at the token limit before the JSON: {why}"
                        if cut_off
                        else why
                    )
                    continue
                preds[record_id] = {
                    "pred": max(probs, key=lambda k: probs[k]),
                    "probs": probs,
                }
            else:
                label, why = parse_label(said, labels)
                if why:
                    failures[record_id] = (
                        f"the answer stopped at the token limit: {why}"
                        if cut_off
                        else why
                    )
                    continue
                preds[record_id] = {"pred": label, "probs": None}
    return preds, failures


def classify(
    rows: list[dict],
    labels: list[str],
    bucket: str,
    distribution: bool = False,
    variant: str = "a",
    run_id: str = "",
) -> Run:
    """Submit a batch job, poll its status and collect its results.

    Use a unique run identifier for the job name and S3 input and output prefixes.
    """
    run_id = run_id or new_run_id()
    s3 = s3_client()
    in_uri = write_manifest(
        rows,
        labels,
        bucket,
        f"input/{run_id}/manifest.jsonl",
        s3,
        distribution,
        variant,
    )
    out_uri = f"s3://{bucket}/output/{run_id}/"
    bedrock = get_bedrock()
    # roleArn is required on AWS (the service role Bedrock assumes to reach S3);
    # the local shim ignores it. BATCH_MODEL lets the cloud round pick a
    # batch-capable model without changing the seam's default.
    job = bedrock.create_model_invocation_job(
        jobName=f"classify-{run_id}",
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
        details = bedrock.get_model_invocation_job(jobIdentifier=arn)
        status = details["status"]
        if status in TERMINAL:
            break
        time.sleep(10)
    counts = {
        key: details[key]
        for key in (
            "totalRecordCount",
            "processedRecordCount",
            "successRecordCount",
            "errorRecordCount",
        )
        if isinstance(details.get(key), int)
    }
    run = Run(status=status, message=str(details.get("message") or ""), counts=counts)
    print(f"job {status}" + (f": {run.message}" if run.message else ""))
    if not run.usable:
        # Return the job failure without collecting predictions.
        return run
    run.predictions, run.failures = collect(
        bucket, out_uri, arn, labels, s3, distribution
    )
    return run


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
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="where to write the predictions; defaults to <dataset>.predictions.jsonl",
    )
    a = p.parse_args()

    rows = [
        json.loads(line)
        for line in (a.data / f"{a.dataset}.jsonl").read_text().splitlines()
    ]
    if a.limit:
        rows = rows[: a.limit]
    labels = json.loads((a.data / f"{a.dataset}.labels.json").read_text())

    run_id = new_run_id()
    run = classify(rows, labels, a.bucket, a.probs, a.variant, run_id)
    if not run.usable:
        # Preserve existing predictions when this run fails.
        raise SystemExit(
            f"job {run.status}: nothing written. "
            f"{a.dataset}.predictions.jsonl is untouched."
        )

    out = a.out or a.data / f"{a.dataset}.predictions.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            got = run.predictions.get(r["id"])
            record = {
                "id": r["id"],
                "label": r["label"],
                "pred": got["pred"] if got else "",
                "run_id": run_id,
            }
            if got and got.get("probs"):
                record["probs"] = got["probs"]
            if not got:
                # Store the error reason for records without a prediction.
                record["failed"] = run.failures.get(
                    r["id"], "the job returned no record for this conversation"
                )
            f.write(json.dumps(record) + "\n")

    hits = sum(run.predictions.get(r["id"], {}).get("pred") == r["label"] for r in rows)
    missing = sum(1 for r in rows if r["id"] not in run.predictions)
    print(
        f"{hits}/{len(rows)} correct, {missing} without a prediction "
        f"({len(run.failures)} records failed or unreadable) -> {out}"
    )
    if run.counts:
        print(
            "  the job reported " + ", ".join(f"{k}={v}" for k, v in run.counts.items())
        )
    if run.status == "PartiallyCompleted":
        print("  the job partially completed: not every record was processed in time")


if __name__ == "__main__":
    main()
