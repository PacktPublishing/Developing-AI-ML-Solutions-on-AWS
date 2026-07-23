# /// script
# dependencies = ["starlette", "uvicorn", "boto3", "pyarrow", "pyiceberg[s3fs]"]
# ///
"""A local SageMaker Feature Store: both planes on one endpoint.

The sagemaker control plane (CreateFeatureGroup, DescribeFeatureGroup) is
AWS json protocol, dispatched on the X-Amz-Target header. The
sagemaker-featurestore-runtime data plane (PutRecord, GetRecord) is
rest-json on /FeatureGroup/{name}. One Starlette app serves both, so
pointing both boto3 clients at this endpoint is the only change between
local and AWS. Backing engines: DynamoDB Local (metadata + online store)
and the Iceberg REST catalog (offline store).

Usage:
  uv run sagemaker-local/app.py          # serves on :8007
"""

import asyncio
import contextlib
import json
import os

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from store import FeatureStore

# -------------------------------------------------------------------------------
# Store and offline buffer
# -------------------------------------------------------------------------------
store = FeatureStore()
FLUSH_SECONDS = float(os.environ.get("OFFLINE_FLUSH_SECONDS", "15"))
offline_queue: asyncio.Queue = asyncio.Queue()


# -------------------------------------------------------------------------------
# Offline flush
# -------------------------------------------------------------------------------
async def flush() -> None:
    """Drain the buffer and commit it to the offline store in one batch."""
    batch = []
    while not offline_queue.empty():
        batch.append(offline_queue.get_nowait())
    if batch:
        await asyncio.to_thread(store.offline_flush, batch)


async def flusher() -> None:
    """Flush the offline buffer on an interval, like the real ingest leg."""
    while True:
        await asyncio.sleep(FLUSH_SECONDS)
        await flush()


@contextlib.asynccontextmanager
async def lifespan(app):
    """Run the flusher for the app's lifetime; drain the buffer on shutdown."""
    task = asyncio.create_task(flusher())
    yield
    task.cancel()
    await flush()


# -------------------------------------------------------------------------------
# Control plane
# -------------------------------------------------------------------------------
async def control_plane(request: Request) -> JSONResponse:
    """Dispatch sagemaker control-plane operations by X-Amz-Target."""
    target = request.headers.get("x-amz-target", "")
    body = json.loads(await request.body() or "{}")
    if target.endswith("CreateFeatureGroup"):
        return JSONResponse(store.create_feature_group(body))
    if target.endswith("DescribeFeatureGroup"):
        try:
            return JSONResponse(store.describe_feature_group(body["FeatureGroupName"]))
        except KeyError:
            return JSONResponse(
                {
                    "__type": "ResourceNotFound",
                    "Message": body.get("FeatureGroupName", ""),
                },
                status_code=400,
            )
    return JSONResponse(
        {"__type": "UnknownOperationException", "Message": target}, status_code=400
    )


# -------------------------------------------------------------------------------
# Data plane
# -------------------------------------------------------------------------------
async def batch_get_record(request: Request) -> JSONResponse:
    """Serve POST /BatchGetRecord."""
    body = json.loads(await request.body())
    return JSONResponse(store.batch_get(body.get("Identifiers", [])))


async def feature_group(request: Request) -> JSONResponse:
    """Serve PUT, GET, and DELETE on /FeatureGroup/{name}."""
    name = request.path_params["name"]
    if request.method == "DELETE":
        record_id = request.query_params.get("RecordIdentifierValueAsString", "")
        if not request.query_params.get("EventTime"):
            return JSONResponse({"Message": "EventTime is required"}, status_code=400)
        try:
            store.delete_record(name, record_id)
        except KeyError:
            return JSONResponse({"Message": f"not found: {name}"}, status_code=404)
        return JSONResponse({})
    if request.method == "PUT":
        body = json.loads(await request.body())
        try:
            values = store.put_record(name, body["Record"])
        except KeyError:
            return JSONResponse({"Message": f"not found: {name}"}, status_code=404)
        if values is not None:
            offline_queue.put_nowait((name, values))
            if FLUSH_SECONDS == 0:  # sync mode for deterministic tests
                await flush()
        return JSONResponse({})
    # GET
    record_id = request.query_params.get("RecordIdentifierValueAsString", "")
    features = request.query_params.getlist("FeatureName") or None
    if record := store.get_record(name, record_id, features):
        return JSONResponse({"Record": record})
    else:
        return JSONResponse({"Message": "record not found"}, status_code=404)


# -------------------------------------------------------------------------------
# App and entrypoint
# -------------------------------------------------------------------------------
app = Starlette(
    lifespan=lifespan,
    routes=[
        Route("/", control_plane, methods=["POST"]),
        Route("/FeatureGroup/{name}", feature_group, methods=["PUT", "GET", "DELETE"]),
        Route("/BatchGetRecord", batch_get_record, methods=["POST"]),
    ],
)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8007, log_level="warning")
