"""Tests for the local Feature Store shim: both API planes and the flush logic."""

import uuid

import app as shim
from starlette.testclient import TestClient

client = TestClient(shim.app)


# -------------------------------------------------------------------------------
# Request helpers
# -------------------------------------------------------------------------------
def make_group(offline: bool = False) -> str:
    """Create a uniquely named feature group through the control plane."""
    name = f"t-{uuid.uuid4().hex[:8]}"
    body = {
        "FeatureGroupName": name,
        "RecordIdentifierFeatureName": "id",
        "EventTimeFeatureName": "ts",
        "FeatureDefinitions": [
            {"FeatureName": n, "FeatureType": "String"} for n in ("id", "score", "ts")
        ],
        "OnlineStoreConfig": {"EnableOnlineStore": True},
    }
    if offline:
        body["OfflineStoreConfig"] = {"S3StorageConfig": {"S3Uri": "s3://x/"}}
    r = client.post(
        "/", json=body, headers={"X-Amz-Target": "SageMaker.CreateFeatureGroup"}
    )
    assert r.status_code == 200
    return name


def put(name: str, rid: str, score: str = "700") -> None:
    """Put one record through the data plane."""
    r = client.put(
        f"/FeatureGroup/{name}",
        json={
            "Record": [
                {"FeatureName": "id", "ValueAsString": rid},
                {"FeatureName": "score", "ValueAsString": score},
                {"FeatureName": "ts", "ValueAsString": "2026-07-19T00:00:00Z"},
            ]
        },
    )
    assert r.status_code == 200


# -------------------------------------------------------------------------------
# Control plane tests
# -------------------------------------------------------------------------------
def test_create_and_describe():
    """The control plane stores and returns the group definition."""
    name = make_group()
    r = client.post(
        "/",
        json={"FeatureGroupName": name},
        headers={"X-Amz-Target": "SageMaker.DescribeFeatureGroup"},
    )
    d = r.json()
    assert d["FeatureGroupStatus"] == "Created"
    assert d["RecordIdentifierFeatureName"] == "id"
    assert len(d["FeatureDefinitions"]) == 3


def test_describe_unknown_group_is_client_error():
    """Describing a missing group returns an AWS-shaped error."""
    r = client.post(
        "/",
        json={"FeatureGroupName": "t-nope"},
        headers={"X-Amz-Target": "SageMaker.DescribeFeatureGroup"},
    )
    assert r.status_code == 400
    assert r.json()["__type"] == "ResourceNotFound"


# -------------------------------------------------------------------------------
# Data plane tests
# -------------------------------------------------------------------------------
def test_put_get_roundtrip():
    """A put record comes back by key with string values."""
    name = make_group()
    put(name, "1", "715")
    r = client.get(
        f"/FeatureGroup/{name}", params={"RecordIdentifierValueAsString": "1"}
    )
    values = {f["FeatureName"]: f["ValueAsString"] for f in r.json()["Record"]}
    assert values["score"] == "715"


def test_get_with_feature_filter():
    """FeatureName query params restrict the returned features."""
    name = make_group()
    put(name, "1")
    r = client.get(
        f"/FeatureGroup/{name}",
        params=[("RecordIdentifierValueAsString", "1"), ("FeatureName", "score")],
    )
    assert [f["FeatureName"] for f in r.json()["Record"]] == ["score"]


def test_missing_record_and_group_are_404():
    """Unknown record and unknown group both return 404 on the data plane."""
    name = make_group()
    assert (
        client.get(
            f"/FeatureGroup/{name}", params={"RecordIdentifierValueAsString": "9"}
        ).status_code
        == 404
    )
    r = client.put(
        "/FeatureGroup/t-nope",
        json={"Record": [{"FeatureName": "id", "ValueAsString": "1"}]},
    )
    assert r.status_code == 404


def test_batch_get_returns_hits_only():
    """BatchGetRecord returns found records and omits misses."""
    name = make_group()
    put(name, "1")
    put(name, "2")
    r = client.post(
        "/BatchGetRecord",
        json={
            "Identifiers": [
                {
                    "FeatureGroupName": name,
                    "RecordIdentifiersValueAsString": ["1", "2", "9"],
                }
            ]
        },
    )
    ids = sorted(x["RecordIdentifierValueAsString"] for x in r.json()["Records"])
    assert ids == ["1", "2"]


def test_delete_requires_event_time_and_removes():
    """DeleteRecord validates EventTime, then removes from both read paths."""
    name = make_group()
    put(name, "1")
    assert (
        client.delete(
            f"/FeatureGroup/{name}", params={"RecordIdentifierValueAsString": "1"}
        ).status_code
        == 400
    )
    assert (
        client.delete(
            f"/FeatureGroup/{name}",
            params={
                "RecordIdentifierValueAsString": "1",
                "EventTime": "2026-07-19T00:00:01Z",
            },
        ).status_code
        == 200
    )
    assert (
        client.get(
            f"/FeatureGroup/{name}", params={"RecordIdentifierValueAsString": "1"}
        ).status_code
        == 404
    )
    r = client.post(
        "/BatchGetRecord",
        json={
            "Identifiers": [
                {"FeatureGroupName": name, "RecordIdentifiersValueAsString": ["1"]}
            ]
        },
    )
    assert r.json()["Records"] == []


# -------------------------------------------------------------------------------
# Offline flush
# -------------------------------------------------------------------------------
def test_offline_flush_batches_per_group(monkeypatch):
    """One flush commits one batch per feature group."""
    calls = []
    monkeypatch.setattr(
        shim.store,
        "_offline_append",
        lambda name, rows: calls.append((name, len(rows))),
    )
    shim.store.offline_flush(
        [("g1", {"id": "1"}), ("g1", {"id": "2"}), ("g2", {"id": "3"})]
    )
    assert sorted(calls) == [("g1", 2), ("g2", 1)]
