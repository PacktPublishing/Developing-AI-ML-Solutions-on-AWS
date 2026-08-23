# /// script
# dependencies = ["pandas", "boto3"]
# ///
"""Read SageMaker Data Capture records into the frames the monitors expect.

A real-time endpoint with DataCaptureConfig logs every request and response to S3 as
JSON Lines, one record per invocation. Each record's endpointInput.data is the CSV the
caller sent (headerless FEATURES rows) and endpointOutput.data is the probability the
scorecard returned per row. read_capture reconstructs the current feature frame and the
live scores from those records, so the same monitors (monitor.run) that watch a batch
watch live traffic instead, with no rescoring.

Usage:
  from capture import read_capture
  current, scores = read_capture("s3://bucket/capture/.../")  # or a local dir or file
"""

import base64
import io
import json
from pathlib import Path

import pandas as pd
from model import CATEGORICAL, FEATURES


def _decode(part: dict) -> str:
    """Return a capture part's payload as text, decoding base64 when that is the encoding."""
    data = part["data"]
    if part.get("encoding") == "BASE64":
        data = base64.b64decode(data).decode("utf-8")
    return data


def _lines(source: str) -> list[str]:
    """Return the JSONL lines under a local dir/file or an s3:// prefix."""
    if source.startswith("s3://"):
        import boto3

        bucket, _, prefix = source[len("s3://") :].partition("/")
        s3 = boto3.client("s3")
        lines: list[str] = []
        for page in s3.get_paginator("list_objects_v2").paginate(
            Bucket=bucket, Prefix=prefix
        ):
            for obj in page.get("Contents", []):
                if obj["Key"].endswith(".jsonl"):
                    body = s3.get_object(Bucket=bucket, Key=obj["Key"])["Body"].read()
                    lines += body.decode("utf-8").splitlines()
        return lines
    path = Path(source)
    files = sorted(path.rglob("*.jsonl")) if path.is_dir() else [path]
    return [ln for f in files for ln in f.read_text().splitlines() if ln.strip()]


def read_capture(source: str) -> tuple[pd.DataFrame, pd.Series]:
    """Parse Data Capture JSONL into (current features, live scores).

    source is a local directory, a single .jsonl file, or an s3:// prefix. One record
    may carry several rows (a batched request), so inputs and outputs are concatenated
    in capture order; the returned scores align row-for-row with the returned frame.
    """
    frames, scores = [], []
    for line in _lines(source):
        record = json.loads(line)["captureData"]
        rows = pd.read_csv(
            io.StringIO(_decode(record["endpointInput"])), header=None, names=FEATURES
        )
        scores += [
            float(x)
            for x in _decode(record["endpointOutput"]).splitlines()
            if x.strip()
        ]
        frames.append(rows)
    current = pd.concat(frames, ignore_index=True)
    for col in CATEGORICAL:
        current[col] = current[col].astype(str)
    return current, pd.Series(scores, name="score")
