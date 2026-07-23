# /// script
# dependencies = ["boto3"]
# ///
"""Generate a day's worth of nested bureau JSON and write it to the raw zone.

Every line of every file is one bureau response, with a nested score block
and arrays of tradelines and inquiries. Passing --drift adds a couple of
extra fields to the payload, which is what you get when a bureau starts
sending new attributes and nobody downstream is told about it.

Usage:
  uv run generate_bureau_files.py --date 2026-07-17 --records 200
  uv run generate_bureau_files.py --date 2026-07-18 --records 200 --drift
"""

import argparse
import contextlib
import datetime
import json
import os
import random

import boto3

# -------------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------------
BUCKET = os.environ.get("RAW_BUCKET", "bureau-raw")
STATES = ["CA", "TX", "NY", "FL", "WA", "IL"]
PRODUCTS = ["credit_card", "auto_loan", "mortgage", "personal_loan"]


# -------------------------------------------------------------------------------
# Record generation
# -------------------------------------------------------------------------------
def bureau_record(day: str, seq: int, drift: bool) -> dict:
    """Generate a single bureau record for a given day and sequence number."""
    applicant_id = 10_000 + seq
    record = {
        "report_id": f"{day}-{seq:05d}",
        "applicant_id": applicant_id,
        "generated_at": f"{day}T02:{seq % 60:02d}:00Z",
        "score": {
            "value": random.randint(420, 850),
            "model": "bureau-v3",
        },
        "tradelines": [
            {
                "product": random.choice(PRODUCTS),
                "balance": random.randint(0, 40_000),
                "months_on_book": random.randint(1, 240),
                "delinquent": random.random() < 0.08,
            }
            for _ in range(random.randint(1, 5))
        ],
        "inquiries": [
            {"purpose": random.choice(PRODUCTS), "days_ago": random.randint(1, 365)}
            for _ in range(random.randint(0, 3))
        ],
        "address": {"state": random.choice(STATES)},
    }
    if drift:  # the bureau added attributes upstream; nobody warned us
        record["score"]["segment"] = random.choice(["thin_file", "high_util", "clean"])
        record["employment"] = {"status": random.choice(["employed", "self_employed"])}
    return record


# -------------------------------------------------------------------------------
# Entrypoint
# -------------------------------------------------------------------------------
def main() -> None:
    """Generate a day of bureau files and upload them to the raw zone."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--date", default=str(datetime.date.today() - datetime.timedelta(days=1))
    )
    parser.add_argument("--records", type=int, default=200)
    parser.add_argument("--files", type=int, default=10)
    parser.add_argument("--drift", action="store_true")
    args = parser.parse_args()

    random.seed(args.date)  # same date, same files: reruns are reproducible
    s3 = boto3.client("s3")
    with contextlib.suppress(s3.exceptions.BucketAlreadyOwnedByYou):
        s3.create_bucket(Bucket=BUCKET)
    per_file = args.records // args.files
    written = 0
    for part in range(args.files):
        records = [
            bureau_record(args.date, part * per_file + i, args.drift)
            for i in range(per_file)
        ]
        key = f"raw/dt={args.date}/part-{part:04d}.json"
        body = "\n".join(json.dumps(r) for r in records)
        s3.put_object(Bucket=BUCKET, Key=key, Body=body.encode())
        written += len(records)

    print(
        f"Wrote {written} records in {args.files} files to s3://{BUCKET}/raw/dt={args.date}/"
    )


if __name__ == "__main__":
    main()
