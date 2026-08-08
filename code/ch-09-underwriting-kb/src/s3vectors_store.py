"""S3 Vectors store: Amazon S3 Vectors on AWS, a from-source shim locally.

The shim mirrors the s3vectors client's create_vector_bucket / create_index /
put_vectors / query_vectors, backed by a local S3 (S3Proxy): each vector is one
JSON object, and the query is a brute-force cosine scan. Selected by
S3VECTORS_LOCAL, the same way the model seam selects Ollama with BEDROCK_LOCAL.
"""

import json
import math
import os

import boto3

from models import embed

REGION = os.environ.get("BEDROCK_REGION", "us-east-1")


# -------------------------------------------------------------------------------
# Local S3 Vectors shim, backed by S3Proxy
# -------------------------------------------------------------------------------
class LocalS3Vectors:
    """An s3vectors stand-in: vectors as S3 objects, cosine scan for the query."""

    def __init__(self) -> None:
        """Open an S3 client pointed at the local S3Proxy endpoint."""
        self._s3 = boto3.client(
            "s3",
            endpoint_url=os.environ.get("S3_ENDPOINT", "http://localhost:9000"),
            aws_access_key_id=os.environ.get("S3_IDENTITY", "local"),
            aws_secret_access_key=os.environ.get("S3_CREDENTIAL", "localsecret"),
            region_name=REGION,
            config=boto3.session.Config(s3={"addressing_style": "path"}),
        )

    def create_vector_bucket(self, vectorBucketName: str, **_) -> dict:
        """Create the backing S3 bucket if it does not already exist."""
        try:
            self._s3.create_bucket(Bucket=vectorBucketName)
        except self._s3.exceptions.ClientError:
            pass  # already exists
        return {}

    def delete_index(self, vectorBucketName: str, indexName: str, **_) -> dict:
        """Delete every object under the index prefix."""
        token = None
        while True:
            kw = {"Bucket": vectorBucketName, "Prefix": f"{indexName}/"}
            if token:
                kw["ContinuationToken"] = token
            resp = self._s3.list_objects_v2(**kw)
            for obj in resp.get("Contents", []):
                self._s3.delete_object(Bucket=vectorBucketName, Key=obj["Key"])
            if not resp.get("IsTruncated"):
                return {}
            token = resp.get("NextContinuationToken")

    def create_index(
        self, vectorBucketName: str, indexName: str, dimension: int, **_
    ) -> dict:
        """Record the index configuration as an object."""
        self._s3.put_object(
            Bucket=vectorBucketName,
            Key=f"{indexName}/_index.json",
            Body=json.dumps({"dimension": dimension}).encode(),
        )
        return {}

    def put_vectors(
        self, vectorBucketName: str, indexName: str, vectors: list, **_
    ) -> dict:
        """Write each vector as one JSON object under the index prefix."""
        for v in vectors:
            self._s3.put_object(
                Bucket=vectorBucketName,
                Key=f"{indexName}/{v['key']}.json",
                Body=json.dumps(
                    {"data": v["data"]["float32"], "metadata": v["metadata"]}
                ).encode(),
            )
        return {}

    def query_vectors(
        self, vectorBucketName: str, indexName: str, topK: int, queryVector: dict, **_
    ) -> dict:
        """Brute-force cosine scan: return the topK nearest as {key, distance, metadata}."""
        q = queryVector["float32"]
        qnorm = math.sqrt(sum(x * x for x in q)) or 1.0
        scored = []
        token = None
        while True:
            kw = {"Bucket": vectorBucketName, "Prefix": f"{indexName}/"}
            if token:
                kw["ContinuationToken"] = token
            resp = self._s3.list_objects_v2(**kw)
            for obj in resp.get("Contents", []):
                if obj["Key"].endswith("/_index.json"):
                    continue
                body = json.loads(
                    self._s3.get_object(Bucket=vectorBucketName, Key=obj["Key"])[
                        "Body"
                    ].read()
                )
                data = body["data"]
                dot = sum(a * b for a, b in zip(q, data))
                dnorm = math.sqrt(sum(x * x for x in data)) or 1.0
                distance = 1 - dot / (qnorm * dnorm)
                scored.append(
                    {
                        "key": obj["Key"],
                        "distance": distance,
                        "metadata": body["metadata"],
                    }
                )
            if not resp.get("IsTruncated"):
                break
            token = resp.get("NextContinuationToken")
        scored.sort(key=lambda r: r["distance"])
        return {"vectors": scored[:topK], "distanceMetric": "cosine"}


def _client():
    """Return an s3vectors client: real boto3, or the local S3Proxy-backed shim."""
    if os.environ.get("S3VECTORS_LOCAL") == "1":
        return LocalS3Vectors()
    return boto3.client("s3vectors", region_name=REGION)


# -------------------------------------------------------------------------------
# The store
# -------------------------------------------------------------------------------
class S3VectorsStore:
    """A vector index in S3 Vectors: cosine k-NN over the embedded memo corpus."""

    def __init__(self) -> None:
        """Open the client and read the bucket and index names from the environment."""
        self.client = _client()
        self.bucket = os.environ.get("S3VECTORS_BUCKET", "ch09-memos")
        self.index = "memo_chunks"
        self._buffer: list[dict] = []

    def reset(self, dim: int) -> None:
        """Create the vector bucket and an empty cosine index of width `dim`."""
        self.client.create_vector_bucket(vectorBucketName=self.bucket)
        try:
            self.client.delete_index(vectorBucketName=self.bucket, indexName=self.index)
        except Exception:
            pass  # no index yet
        self.client.create_index(
            vectorBucketName=self.bucket,
            indexName=self.index,
            dimension=dim,
            distanceMetric="cosine",
            dataType="float32",
            metadataConfiguration={"nonFilterableMetadataKeys": ["content"]},
        )

    def add(
        self,
        loan_id: int,
        borrower: str,
        chunk_index: int,
        content: str,
        vector: list[float],
    ) -> None:
        """Buffer one embedded chunk, flushing to S3 Vectors in batches."""
        self._buffer.append(
            {
                "key": f"{loan_id}-{chunk_index}",
                "data": {"float32": [float(x) for x in vector]},
                "metadata": {
                    "loan_id": loan_id,
                    "borrower": borrower,
                    "chunk_index": chunk_index,
                    "content": content,
                },
            }
        )
        if len(self._buffer) >= 200:
            self._flush()

    def _flush(self) -> None:
        """Write the buffered vectors and clear the buffer."""
        if self._buffer:
            self.client.put_vectors(
                vectorBucketName=self.bucket, indexName=self.index, vectors=self._buffer
            )
            self._buffer = []

    def finalize(self) -> None:
        """Flush any remaining buffered vectors."""
        self._flush()

    def search(
        self, runtime, query: str, k: int = 5
    ) -> list[tuple[int, str, str, float]]:
        """Return the k nearest chunks as (loan_id, borrower, content, similarity)."""
        vector = embed(runtime, [query])[0]
        resp = self.client.query_vectors(
            vectorBucketName=self.bucket,
            indexName=self.index,
            topK=k,
            queryVector={"float32": [float(x) for x in vector]},
            returnMetadata=True,
            returnDistance=True,
        )
        hits = []
        for v in resp["vectors"]:
            md = v["metadata"]
            similarity = 1 - v["distance"]  # cosine distance to similarity
            hits.append((md["loan_id"], md["borrower"], md["content"], similarity))
        return hits
