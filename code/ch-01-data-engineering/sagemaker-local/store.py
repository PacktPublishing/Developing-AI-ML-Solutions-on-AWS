"""Backing store for the local Feature Store shim.

Metadata and the online store live in DynamoDB Local; the offline store,
when a group enables it, is an Iceberg table in the local REST catalog.
All feature values are kept as strings, which is faithful: the real
featurestore-runtime speaks ValueAsString on both Put and Get.
"""

import json
import os

import boto3

# -------------------------------------------------------------------------------
# DynamoDB connection
# -------------------------------------------------------------------------------
META_TABLE = "sagemaker-feature-groups"


def _dynamodb():
    """Create a DynamoDB resource connected to the local endpoint."""
    return boto3.resource(
        "dynamodb",
        endpoint_url=os.environ.get("ONLINE_STORE_URL", "http://localhost:8000"),
        region_name="us-east-1",
        aws_access_key_id="local",
        aws_secret_access_key="localsecret",
    )


# -------------------------------------------------------------------------------
# Feature store
# -------------------------------------------------------------------------------
class FeatureStore:
    """Feature-group metadata, the online store, and offline appends."""

    def __init__(self):
        """Connect to DynamoDB Local and ensure the metadata table exists."""
        self.ddb = _dynamodb()
        self.meta = self._ensure_table(META_TABLE, "feature_group_name")

    def _ensure_table(self, name: str, key: str):
        """Create the DynamoDB table if missing and return it."""
        try:
            self.ddb.create_table(
                TableName=name,
                KeySchema=[{"AttributeName": key, "KeyType": "HASH"}],
                AttributeDefinitions=[{"AttributeName": key, "AttributeType": "S"}],
                BillingMode="PAY_PER_REQUEST",
            ).wait_until_exists()
        except self.ddb.meta.client.exceptions.ResourceInUseException:
            pass
        return self.ddb.Table(name)

    # -------------------------------------------------------------------------------
    # Control plane
    # -------------------------------------------------------------------------------

    def create_feature_group(self, config: dict) -> dict:
        """Register a group and provision its online table."""
        name = config["FeatureGroupName"]
        self.meta.put_item(
            Item={
                "feature_group_name": name,
                "record_identifier": config["RecordIdentifierFeatureName"],
                "event_time": config["EventTimeFeatureName"],
                "feature_definitions": json.dumps(config["FeatureDefinitions"]),
                "online": bool(
                    config.get("OnlineStoreConfig", {}).get("EnableOnlineStore")
                ),
                "offline": "OfflineStoreConfig" in config,
            }
        )
        self._ensure_table(f"fg-{name}", "record_id")
        return {
            "FeatureGroupArn": f"arn:aws:sagemaker:local:000000000000:feature-group/{name}"
        }

    def describe_feature_group(self, name: str) -> dict:
        """Return the stored definition in DescribeFeatureGroup shape."""
        item = self.meta.get_item(Key={"feature_group_name": name}).get("Item")
        if item is None:
            raise KeyError(name)
        return {
            "FeatureGroupArn": f"arn:aws:sagemaker:local:000000000000:feature-group/{name}",
            "FeatureGroupName": name,
            "RecordIdentifierFeatureName": item["record_identifier"],
            "EventTimeFeatureName": item["event_time"],
            "FeatureDefinitions": json.loads(item["feature_definitions"]),
            "FeatureGroupStatus": "Created",
            "OnlineStoreConfig": {"EnableOnlineStore": bool(item["online"])},
        }

    # -------------------------------------------------------------------------------
    # Data plane
    # -------------------------------------------------------------------------------

    def put_record(self, name: str, record: list[dict]) -> dict | None:
        """Write a record to the online store; return values if offline-bound."""
        item = self.meta.get_item(Key={"feature_group_name": name}).get("Item")
        if item is None:
            raise KeyError(name)
        values = {f["FeatureName"]: f["ValueAsString"] for f in record}
        row = {"record_id": values[item["record_identifier"]], **values}
        self.ddb.Table(f"fg-{name}").put_item(Item=row)
        return values if item["offline"] else None

    def get_record(
        self, name: str, record_id: str, features: list[str] | None
    ) -> list[dict]:
        """Read one record by identifier, optionally filtered to features."""
        got = (
            self.ddb.Table(f"fg-{name}")
            .get_item(Key={"record_id": record_id})
            .get("Item")
        )
        if not got:
            return []
        got.pop("record_id", None)
        keep = features or list(got)
        return [
            {"FeatureName": k, "ValueAsString": str(v)}
            for k, v in got.items()
            if k in keep
        ]

    def delete_record(self, name: str, record_id: str) -> None:
        """Remove a record from the online store."""
        item = self.meta.get_item(Key={"feature_group_name": name}).get("Item")
        if item is None:
            raise KeyError(name)
        self.ddb.Table(f"fg-{name}").delete_item(Key={"record_id": record_id})

    def batch_get(self, identifiers: list[dict]) -> dict:
        """Serve BatchGetRecord across groups and identifiers."""
        records, errors = [], []
        for ident in identifiers:
            name = ident["FeatureGroupName"]
            features = ident.get("FeatureNames")
            for rid in ident["RecordIdentifiersValueAsString"]:
                try:
                    rec = self.get_record(name, rid, features)
                except Exception:
                    errors.append(
                        {
                            "FeatureGroupName": name,
                            "RecordIdentifierValueAsString": rid,
                            "ErrorCode": "ResourceNotFound",
                            "ErrorMessage": name,
                        }
                    )
                    continue
                if rec:
                    records.append(
                        {
                            "FeatureGroupName": name,
                            "RecordIdentifierValueAsString": rid,
                            "Record": rec,
                        }
                    )
        return {"Records": records, "Errors": errors, "UnprocessedIdentifiers": []}

    # -------------------------------------------------------------------------------
    # Offline store
    # -------------------------------------------------------------------------------
    def offline_flush(self, batch: list[tuple[str, dict]]) -> None:
        """Append a batch of buffered rows, one commit per feature group."""
        from collections import defaultdict

        groups: dict[str, list[dict]] = defaultdict(list)
        for name, values in batch:
            groups[name].append(values)
        for name, rows in groups.items():
            self._offline_append(name, rows)

    def _offline_append(self, name: str, rows: list[dict]) -> None:
        """Append rows to the group's offline Iceberg table in one commit."""
        # offline store: an Iceberg table per group, all-string columns, the
        # same open format a real Iceberg-format offline store uses
        import pyarrow as pa
        from pyiceberg.catalog import load_catalog
        from pyiceberg.exceptions import (
            NamespaceAlreadyExistsError,
            NoSuchTableError,
        )

        catalog = load_catalog(
            "local",
            uri=os.environ.get("ICEBERG_REST_URI", "http://localhost:8181"),
            warehouse="s3://feature-store/",
            **{
                "s3.endpoint": os.environ.get(
                    "AWS_ENDPOINT_URL", "http://localhost:9000"
                ),
                "s3.access-key-id": "local",
                "s3.secret-access-key": "localsecret",
                "s3.region": "us-east-1",
            },
        )
        try:
            catalog.create_namespace("offline")
        except NamespaceAlreadyExistsError:
            pass
        columns = sorted(rows[0])
        data = pa.table(
            {c: pa.array([r.get(c, "") for r in rows], pa.string()) for c in columns}
        )
        table_name = f"offline.{name.replace('-', '_')}"
        try:
            table = catalog.load_table(table_name)
        except NoSuchTableError:
            table = catalog.create_table(table_name, schema=data.schema)
        table.append(data)
