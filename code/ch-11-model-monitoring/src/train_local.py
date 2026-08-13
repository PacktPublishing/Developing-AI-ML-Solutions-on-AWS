# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["sagemaker>=3,<4", "sagemaker-mlops", "botocore[crt]", "boto3"]
# ///
"""Train the scorecard in SageMaker local mode (Mode.LOCAL_CONTAINER).

The same image and /opt/ml contract as the managed training job, but the SDK runs the
container locally instead of a bare docker run, so the training API is exactly the one the
cloud job uses. Produces runs-local/model/scorecard.cbm.

Env: IMAGE (default ch11-scorecard:latest).
"""

import os
import shutil
import tarfile
from pathlib import Path

from sagemaker.train import ModelTrainer
from sagemaker.train.configs import Compute, InputData
from sagemaker.train.model_trainer import Mode

ROOT = Path(__file__).resolve().parent.parent
IMAGE = os.environ.get("IMAGE", "ch11-scorecard:latest")
REFERENCE = ROOT / "runs-local" / "input" / "data" / "reference"
MODEL_DIR = ROOT / "runs-local" / "model"

# The SDK bind-mounts the container's /opt/ml/model to <root>/model; keep <root> under the
# repo (Docker Desktop shares /Users, not the default /var/folders TMPDIR) so the mount is
# real and the artifact collection sees the model. A local path data source with
# instance_type=local is mounted straight in, so training needs no S3.
staging = ROOT / "runs-local" / ".sm-train"
shutil.rmtree(staging, ignore_errors=True)
staging.mkdir(parents=True)
trainer = ModelTrainer(
    training_image=IMAGE,
    training_mode=Mode.LOCAL_CONTAINER,
    compute=Compute(instance_type="local", instance_count=1),
    hyperparameters={"iterations": "300", "depth": "5", "learning_rate": "0.1"},
    local_container_root=str(staging),
)
trainer.train(
    input_data_config=[InputData(channel_name="reference", data_source=str(REFERENCE))],
    wait=True,
)

# the SDK collects /opt/ml/model into compressed_artifacts/model.tar.gz; unpack the scorecard
# into runs-local/model, where the monitor and the pipeline read it
MODEL_DIR.mkdir(parents=True, exist_ok=True)
with tarfile.open(staging / "compressed_artifacts" / "model.tar.gz") as tar:
    tar.extractall(MODEL_DIR)
shutil.rmtree(staging, ignore_errors=True)
print("model ->", MODEL_DIR, sorted(p.name for p in MODEL_DIR.iterdir()))
