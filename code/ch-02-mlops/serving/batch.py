# /// script
# requires-python = ">=3.10"
# dependencies = ["boto3", "pandas"]
# ///
"""Batch-score a file through the BYOC serverless endpoint.

Batch transform on SageMaker is an instance-based job; on a free-tier account
with zero instance quota it will not run. The serverless equivalent, and the one
this chapter uses, is to stream the file through the same serverless endpoint
that serves real time — one container image, one model, whether a request is a
single application or a nightly file of them. The endpoint scales to zero between
runs, so an occasional batch costs only the seconds it runs.

The same script scores against a local container (pass --url) so batch scoring is
identical in both worlds.

Usage:
  uv run serving/batch.py --endpoint ch02-challenger-byoc --input data/split/test.csv --output /tmp/scored.csv
  uv run serving/batch.py --url http://localhost:8092 --input data/split/test.csv --output /tmp/scored.csv
"""

import argparse
import json
import os

import pandas as pd


def _score_chunk(
    records: list[dict], endpoint: str | None, url: str | None, region: str
):
    """Score one chunk of application records, returning a list of PDs."""
    body = json.dumps(records).encode("utf-8")
    if url:
        import urllib.request

        req = urllib.request.Request(
            f"{url}/invocations",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())["pd"]
    import boto3

    rt = boto3.client("sagemaker-runtime", region_name=region)
    resp = rt.invoke_endpoint(
        EndpointName=endpoint, ContentType="application/json", Body=body
    )
    return json.loads(resp["Body"].read())["pd"]


def main() -> None:
    """Read the input file, score it in chunks, write input + a pd column."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", help="SageMaker serverless endpoint name")
    ap.add_argument(
        "--url", help="local container base URL, e.g. http://localhost:8092"
    )
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--chunk", type=int, default=500)
    ap.add_argument(
        "--region", default=os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    )
    args = ap.parse_args()
    if not (args.endpoint or args.url):
        raise SystemExit("pass --endpoint (AWS) or --url (local)")

    df = pd.read_csv(args.input)
    pds: list[float] = []
    for start in range(0, len(df), args.chunk):
        chunk = df.iloc[start : start + args.chunk]
        records = chunk.to_dict(orient="records")
        pds.extend(_score_chunk(records, args.endpoint, args.url, args.region))
        print(f"scored {min(start + args.chunk, len(df))}/{len(df)}")

    out = df.copy()
    out["pd"] = pds
    out.to_csv(args.output, index=False)
    hi = sum(p >= 0.5 for p in pds)
    print(
        f"wrote {args.output}: {len(pds)} rows, {hi} scored PD>=0.5, mean PD {sum(pds) / len(pds):.4f}"
    )


if __name__ == "__main__":
    main()
