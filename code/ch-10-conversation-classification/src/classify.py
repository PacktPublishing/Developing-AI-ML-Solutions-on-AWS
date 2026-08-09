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
import time
from pathlib import Path

from batch import get_bedrock, s3_client
from models import TEXT_MODEL, answer_text

INSTRUCTION = (
    "You are a bank's conversation router. Classify the conversation into exactly"
    " one category from the list. Reply with only the category name, nothing else."
)


def model_input(text: str, labels: list[str]) -> dict:
    """Build the Converse request body that classifies one conversation."""
    prompt = (
        f"{INSTRUCTION}\n\nCategories:\n- "
        + "\n- ".join(labels)
        + f"\n\nConversation:\n{text}\n\nCategory:"
    )
    return {
        "messages": [{"role": "user", "content": [{"text": prompt}]}],
        "inferenceConfig": {"maxTokens": 32, "temperature": 0},
    }


def write_manifest(
    rows: list[dict], labels: list[str], bucket: str, key: str, s3
) -> str:
    """Write the {recordId, modelInput} JSONL manifest to S3; return its s3 URI."""
    lines = [
        json.dumps({"recordId": r["id"], "modelInput": model_input(r["text"], labels)})
        for r in rows
    ]
    s3.put_object(Bucket=bucket, Key=key, Body="\n".join(lines).encode("utf-8"))
    return f"s3://{bucket}/{key.rsplit('/', 1)[0]}/"


def collect(
    bucket: str, out_prefix: str, job_arn: str, labels: list[str], s3
) -> dict[str, str]:
    """Read the .jsonl.out shards and map each answer to the nearest known label."""
    job_id = job_arn.split("/")[-1]
    prefix = (
        f"{out_prefix.removeprefix('s3://').partition('/')[2].rstrip('/')}/{job_id}/"
    )
    lower = {label.lower(): label for label in labels}
    preds: dict[str, str] = {}
    for obj in s3.list_objects_v2(Bucket=bucket, Prefix=prefix).get("Contents", []):
        if not obj["Key"].endswith(".out"):
            continue
        body = (
            s3.get_object(Bucket=bucket, Key=obj["Key"])["Body"].read().decode("utf-8")
        )
        for line in body.splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            said = answer_text(rec["modelOutput"]).lower()
            match = next((orig for low, orig in lower.items() if low in said), None)
            preds[rec["recordId"]] = match or said
    return preds


def classify(rows: list[dict], labels: list[str], bucket: str) -> dict[str, str]:
    """Run the whole batch flow and return {conversation id: predicted category}."""
    s3 = s3_client()
    in_uri = write_manifest(rows, labels, bucket, "input/manifest.jsonl", s3)
    out_uri = f"s3://{bucket}/output/"
    bedrock = get_bedrock()
    job = bedrock.create_model_invocation_job(
        jobName=f"classify-{int(time.time())}",
        modelId=TEXT_MODEL,
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
    return collect(bucket, out_uri, arn, labels, s3)


def main() -> None:
    """Classify a prepared dataset and write predictions.jsonl."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="conversations")
    p.add_argument("--bucket", default="ch10-batch")
    p.add_argument("--data", type=Path, default=Path("data/generated"))
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()

    rows = [
        json.loads(line)
        for line in (a.data / f"{a.dataset}.jsonl").read_text().splitlines()
    ]
    if a.limit:
        rows = rows[: a.limit]
    labels = json.loads((a.data / f"{a.dataset}.labels.json").read_text())

    preds = classify(rows, labels, a.bucket)
    out = a.data / f"{a.dataset}.predictions.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(
                json.dumps(
                    {"id": r["id"], "label": r["label"], "pred": preds.get(r["id"], "")}
                )
                + "\n"
            )
    hits = sum(preds.get(r["id"]) == r["label"] for r in rows)
    print(f"{hits}/{len(rows)} correct -> {out}")


if __name__ == "__main__":
    main()
