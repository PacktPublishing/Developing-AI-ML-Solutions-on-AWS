# /// script
# dependencies = ["boto3"]
# ///
"""Read one applicant's features back from the online store.

This is the serving-path read: given an applicant id, it fetches the latest
feature row straight from DynamoDB by key, the same way a model would when it
scores a request at inference time.

Usage:
  uv run lookup_features.py --applicant 10073
"""

import argparse
import os

import boto3


def main() -> None:
    """Fetch one applicant's features by key from the online store."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--applicant", type=int, default=10073)
    args = parser.parse_args()

    table = boto3.resource(
        "dynamodb",
        endpoint_url=os.environ.get("ONLINE_STORE_URL", "http://localhost:8000"),
    ).Table("applicant-credit-profile")

    item = table.get_item(Key={"applicant_id": args.applicant}).get("Item")
    if item is None:
        raise SystemExit(f"No features for applicant {args.applicant}")
    print(
        {k: int(v) if k not in ["state", "event_time"] else v for k, v in item.items()}
    )


if __name__ == "__main__":
    main()
