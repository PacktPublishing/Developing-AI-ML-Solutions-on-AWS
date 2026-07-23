# /// script
# dependencies = ["boto3"]
# ///
"""Parity check: the same Feature Store op matrix, shim or AWS.

Runs describe, put x2, get, batch_get, delete, get-after-delete, and
batch-after-delete against whichever endpoint the environment selects,
printing one normalized line per op so two runs can be diffed directly.

Usage:
  FEATURE_STORE_URL=http://localhost:8007 uv run feature-store/parity_check.py
  uv run feature-store/parity_check.py         # real AWS
"""

import datetime
import os
import time

import boto3
from botocore.exceptions import ClientError

# -------------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------------
GROUP = os.environ.get("FEATURE_GROUP", "applicant-credit-profile-fg")
NOW = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
ROWS = {
    "90001": {
        "score": "700",
        "state": "CA",
        "tradeline_count": "3",
        "total_balance": "1000",
        "delinquent_count": "0",
    },
    "90002": {
        "score": "650",
        "state": "TX",
        "tradeline_count": "5",
        "total_balance": "2500",
        "delinquent_count": "1",
    },
}


# -------------------------------------------------------------------------------
# Record helpers
# -------------------------------------------------------------------------------
def record(rid: str) -> list[dict]:
    """Build a PutRecord payload for one row."""
    values = {"applicant_id": rid, **ROWS[rid], "event_time": NOW}
    return [{"FeatureName": k, "ValueAsString": v} for k, v in values.items()]


def normalize(rec: list[dict]) -> dict:
    """Reduce a record list to a sorted feature dict, event_time dropped."""
    out = {f["FeatureName"]: f["ValueAsString"] for f in rec}
    out.pop("event_time", None)  # differs between runs by design
    return dict(sorted(out.items()))


# -------------------------------------------------------------------------------
# Op matrix and entrypoint
# -------------------------------------------------------------------------------
def main() -> None:
    """Run the op matrix and print one line per operation."""
    endpoint = os.environ.get("FEATURE_STORE_URL")
    kwargs = {"endpoint_url": endpoint} if endpoint else {}
    sm = boto3.client("sagemaker", **kwargs)
    rt = boto3.client("sagemaker-featurestore-runtime", **kwargs)

    d = sm.describe_feature_group(FeatureGroupName=GROUP)
    print(
        "describe:",
        d["FeatureGroupStatus"],
        d["RecordIdentifierFeatureName"],
        d["EventTimeFeatureName"],
        len(d["FeatureDefinitions"]),
    )

    for rid in ROWS:
        rt.put_record(FeatureGroupName=GROUP, Record=record(rid))
    print("put:", "ok", len(ROWS))

    got = rt.get_record(FeatureGroupName=GROUP, RecordIdentifierValueAsString="90001")
    print("get:", normalize(got["Record"]))

    batch = rt.batch_get_record(
        Identifiers=[
            {
                "FeatureGroupName": GROUP,
                "RecordIdentifiersValueAsString": ["90001", "90002"],
            }
        ]
    )
    print(
        "batch_get:",
        sorted(
            (r["RecordIdentifierValueAsString"], normalize(r["Record"]))
            for r in batch["Records"]
        ),
    )

    # the tombstone's EventTime must be strictly newer than the record's,
    # or the online store discards the delete as stale
    delete_time = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    rt.delete_record(
        FeatureGroupName=GROUP,
        RecordIdentifierValueAsString="90001",
        EventTime=delete_time,
    )
    print("delete:", "ok")

    # the real online store deletes asynchronously; poll until the record
    # is gone so both worlds report the same eventual state
    status = "still found after 60s"
    for i in range(13):
        try:
            rt.get_record(FeatureGroupName=GROUP, RecordIdentifierValueAsString="90001")
            time.sleep(5)
        except ClientError:
            status = f"gone (after ~{i * 5}s)"
            break
    print("get_after_delete:", status)

    batch = rt.batch_get_record(
        Identifiers=[
            {
                "FeatureGroupName": GROUP,
                "RecordIdentifiersValueAsString": ["90001", "90002"],
            }
        ]
    )
    print(
        "batch_after_delete:",
        sorted(r["RecordIdentifierValueAsString"] for r in batch["Records"]),
        "unprocessed:",
        batch.get("UnprocessedIdentifiers", []),
        "errors:",
        [e.get("ErrorCode") for e in batch.get("Errors", [])],
    )


if __name__ == "__main__":
    main()
