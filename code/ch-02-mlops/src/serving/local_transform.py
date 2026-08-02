# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["sagemaker>=3,<4", "botocore[crt]", "boto3"]
# ///
"""Batch-score a file with a SageMaker Batch Transform job in LOCAL mode.

The managed counterpart of `make batch` (which is a hand loop over /invocations):
this runs a real SageMaker Transform job on the same BYOC container, in local mode
(instance_type="local"). The SDK brings the container up via docker-compose, streams
the input through /invocations, and writes a .out file, exactly as a cloud Transform
job would — swap instance_type to a real one and point at S3 for the cloud. The
chapter runs it locally because a cloud Transform job is instance-based and the
free-tier account has no training/processing instance quota.

The container reads CSV with a header, so the whole file goes as one payload
(split_type=None) rather than line by line. Local mode still resolves the AWS
account and validates the role, so real credentials and a SageMaker role are needed.

Env: MODEL_IMAGE (ecr uri), SAGEMAKER_ROLE_ARN, MODEL_PATH (artifact dir),
     INPUT (default data/split/test.csv).
"""

import glob
import importlib
import os
import tarfile
import tempfile

from sagemaker.core.local import LocalSession
from sagemaker.core.transformer import Transformer

# local mode reads sagemaker.serve.model_builder lazily (core/local/image.py); load
# the submodule up front so the attribute exists when the transform job starts.
importlib.import_module("sagemaker.serve.model_builder")

IMAGE = os.environ["MODEL_IMAGE"]
ROLE = os.environ["SAGEMAKER_ROLE_ARN"]
MODEL_PATH = os.environ.get("MODEL_PATH", "runs-local/model")
INPUT = os.path.abspath(os.environ.get("INPUT", "data/split/test.csv"))

sess = LocalSession()
sess.config = {"local": {"local_code": True}}

# package the trained artifact into model.tar.gz (files at the tar root, which
# SageMaker unpacks into /opt/ml/model where the container loads them)
staging = tempfile.mkdtemp()
tar_path = os.path.join(staging, "model.tar.gz")
with tarfile.open(tar_path, "w:gz") as tar:
    for fname in os.listdir(MODEL_PATH):
        src = os.path.join(MODEL_PATH, fname)
        if os.path.isfile(src):
            tar.add(src, arcname=fname)

MODEL_NAME = "ch02-scorecard-transform"
sess.create_model(
    name=MODEL_NAME,
    role=ROLE,
    # local mode reads primary_container["Environment"] unconditionally, so set it
    container_defs={
        "Image": IMAGE,
        "ModelDataUrl": f"file://{tar_path}",
        "Environment": {},
    },
)

out_dir = tempfile.mkdtemp()
transformer = Transformer(
    model_name=MODEL_NAME,
    instance_count=1,
    instance_type="local",
    strategy="SingleRecord",
    output_path=f"file://{out_dir}",
    accept="text/csv",
    max_payload=100,
    sagemaker_session=sess,
)
# job_name pins the run so the SDK does not call the real DescribeModel; wait=False
# keeps it from polling a real DescribeTransformJob (there is none in local mode).
transformer.transform(
    data=f"file://{INPUT}",
    content_type="text/csv",
    split_type=None,
    job_name="ch02-scorecard-transform-local",
    wait=False,
)

outs = glob.glob(os.path.join(out_dir, "**", "*.out"), recursive=True)
print("output files:", outs)
if outs:
    with open(outs[0]) as f:
        rows = f.read().splitlines()
    print(f"scored {len(rows)} rows; first 3 PDs:", rows[:3])
