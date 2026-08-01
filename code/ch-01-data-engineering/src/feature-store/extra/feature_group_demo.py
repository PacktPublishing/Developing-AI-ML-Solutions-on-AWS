# /// script
# dependencies = ["boto3", "psycopg2-binary"]
# ///
"""The feature group through the real API, local or AWS by endpoint alone.

Creates an online feature group from the gold mart's schema, puts the top
mart rows as records, and reads one back by key. Locally both clients
point at the sagemaker-local shim; on AWS, unset the endpoint and the
same calls hit the real service.

Usage:
  uv run feature-store/extra/feature_group_demo.py --applicant 10073
"""

import argparse
import datetime
import os
import time

import boto3
import psycopg2

# -------------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------------
GROUP = os.environ.get("FEATURE_GROUP", "applicant-credit-profile-fg")
WAREHOUSE_DSN = os.environ.get(
    "WAREHOUSE_DSN", "postgresql://loader:loader@localhost:5439/bureau?sslmode=require"
)
FEATURES = [
    ("applicant_id", "Integral"),
    ("score", "Integral"),
    ("state", "String"),
    ("tradeline_count", "Integral"),
    ("total_balance", "Integral"),
    ("delinquent_count", "Integral"),
    ("event_time", "String"),
]


# -------------------------------------------------------------------------------
# Clients and group setup
# -------------------------------------------------------------------------------
def clients():
    """Build the sagemaker and runtime clients for the selected endpoint."""
    endpoint = os.environ.get("FEATURE_STORE_URL")
    kwargs = {"endpoint_url": endpoint} if endpoint else {}
    return (
        boto3.client("sagemaker", **kwargs),
        boto3.client("sagemaker-featurestore-runtime", **kwargs),
    )


def ensure_group(sm) -> None:
    """Create the feature group if needed and wait until Created."""
    try:
        status = sm.describe_feature_group(FeatureGroupName=GROUP)["FeatureGroupStatus"]
    except Exception:
        sm.create_feature_group(
            FeatureGroupName=GROUP,
            RecordIdentifierFeatureName="applicant_id",
            EventTimeFeatureName="event_time",
            FeatureDefinitions=[
                {"FeatureName": n, "FeatureType": t} for n, t in FEATURES
            ],
            OnlineStoreConfig={"EnableOnlineStore": True},
        )
        status = "Creating"
    while status == "Creating":  # real AWS takes a minute; the shim is instant
        time.sleep(10)
        status = sm.describe_feature_group(FeatureGroupName=GROUP)["FeatureGroupStatus"]
    print(f"feature group {GROUP}: {status}")


# -------------------------------------------------------------------------------
# Warehouse read
# -------------------------------------------------------------------------------
def mart_rows(limit: int) -> list[dict]:
    """Read the top mart rows as string-valued records."""
    with psycopg2.connect(WAREHOUSE_DSN) as conn, conn.cursor() as cur:
        cur.execute(
            "select applicant_id, score, state, tradeline_count,"
            " total_balance, delinquent_count"
            " from bureau_marts.applicant_credit_profile"
            " order by score desc limit %s",
            (limit,),
        )
        rows = cur.fetchall()
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    names = [n for n, _ in FEATURES[:-1]]
    return [dict(zip(names, map(str, r)), event_time=now) for r in rows]


# -------------------------------------------------------------------------------
# Entrypoint
# -------------------------------------------------------------------------------
def main() -> None:
    """Create, fill, and read back the feature group."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--applicant", default="10073")
    parser.add_argument("--records", type=int, default=10)
    args = parser.parse_args()

    sm, runtime = clients()
    ensure_group(sm)

    for row in mart_rows(args.records):
        runtime.put_record(
            FeatureGroupName=GROUP,
            Record=[{"FeatureName": k, "ValueAsString": v} for k, v in row.items()],
        )
    print(f"put {args.records} records")

    got = runtime.get_record(
        FeatureGroupName=GROUP, RecordIdentifierValueAsString=args.applicant
    )
    print({f["FeatureName"]: f["ValueAsString"] for f in got["Record"]})


if __name__ == "__main__":
    main()
