# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["sagemaker>=3,<4", "botocore[crt]", "torch", "torchvision",
#                 "facenet-pytorch", "numpy<2", "scipy<1.14", "pillow", "captum",
#                 "psycopg[binary]", "psycopg-pool", "fastapi", "uvicorn", "requests"]
# ///
"""Serve the chapter's InferenceSpec through ModelBuilder, in process, on the Apple GPU.

PYTHONPATH must include ../src; the Makefile's serve-inprocess target sets it.

  SM_OFFLINE=1 uv run local/serve_inprocess.py [key] [--claim SUBJECT]

Mode.IN_PROCESS is the third of ModelBuilder's deployment modes. SAGEMAKER_ENDPOINT and
LOCAL_CONTAINER both put the model in a container, and Docker passes no Metal device
through, so this is the only one that reaches an Apple GPU. The spec it serves is the one
the SageMaker endpoint runs on AWS.

With no account (SM_OFFLINE=1) the SDK's role validation and default-bucket lookup are
stubbed, exactly as Chapter 4 does for its local container endpoint.
"""

import argparse
import json
import os

from face_spec import FaceEmbeddingSpec
from sagemaker.serve.builder.schema_builder import SchemaBuilder
from sagemaker.serve.mode.function_pointers import Mode
from sagemaker.serve.model_builder import ModelBuilder

ROLE = os.environ.get("SAGEMAKER_ROLE_ARN", "arn:aws:iam::000000000000:role/local")


def main() -> None:
    """Build and deploy the spec in process, then put one request through it."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("key", nargs="?", default="probe/subject_000/selfie.jpg")
    ap.add_argument("--claim", default="subject_000")
    a = ap.parse_args()

    if os.environ.get("SM_OFFLINE") == "1":
        from sagemaker_offline import use_local_stubs

        use_local_stubs()

    payload = {"op": "match", "key": a.key, "claim": a.claim}
    builder = ModelBuilder(
        inference_spec=FaceEmbeddingSpec(),
        schema_builder=SchemaBuilder(
            sample_input=payload, sample_output={"matched": True, "device": "mps"}
        ),
        role_arn=ROLE,
        mode=Mode.IN_PROCESS,
    )
    builder.build()
    endpoint = builder.deploy_local(wait=True)
    try:
        # the same invoke() shape Chapter 4 uses against its local container endpoint
        response = endpoint.invoke(
            body=json.dumps(payload), content_type="application/json"
        )
        # the endpoint's deserializer has already parsed the body into a dict
        body = getattr(response, "body", response)
        result = json.loads(body) if isinstance(body, str | bytes) else body
        print(
            f"matched={result.get('matched')} identified={result.get('identified')} "
            f"device={result.get('device')}"
        )
    finally:
        builder.modes[str(Mode.IN_PROCESS)].destroy_server()


if __name__ == "__main__":
    main()
