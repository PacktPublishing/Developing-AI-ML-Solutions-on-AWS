# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["sagemaker>=3,<4", "sagemaker-mlops", "botocore[crt]", "boto3"]
# ///
"""Train a model in SageMaker local mode (Mode.LOCAL_CONTAINER).

The same image and /opt/ml contract as the managed training job, but the SDK runs the
container locally instead of a bare docker run, so the training API is exactly the one the
cloud job uses. MODEL selects the incumbent scorecard or the CatBoost challenger.

Env: MODEL (scorecard|challenger, default scorecard), IMAGE (default ch02-<model>:latest).
"""

import os
import shutil
import tarfile
from pathlib import Path

from sagemaker.train import ModelTrainer
from sagemaker.train.configs import Compute, InputData
from sagemaker.train.model_trainer import Mode

ROOT = Path(__file__).resolve().parent.parent
MODEL = os.environ.get("MODEL", "scorecard")
IMAGE = os.environ.get("IMAGE", f"ch02-{MODEL}:latest")
MODEL_DIR = (
    ROOT / "runs-local" / ("model" if MODEL == "scorecard" else "model-challenger")
)
CHANNELS = ROOT / "runs-local" / "input" / "data"
# the shared MLflow tracking store: sequential MLFLOW runs seed and re-save it, so both
# models land in one sqlite database, the way the mounted /mlflow-store used to hold them
SHARED_DB = ROOT / "mlflow-store" / "mlruns.db"

# Keep the SDK's container root under the repo: Docker Desktop shares /Users, not the default
# /var/folders TMPDIR, so a temp dir there would bind-mount empty and lose the model. A local
# path data source with instance_type=local is mounted straight in, so training needs no S3.
staging = ROOT / "runs-local" / f".sm-train-{MODEL}"
shutil.rmtree(staging, ignore_errors=True)
staging.mkdir(parents=True)

# MLFLOW=1 logs the run to the serverless MLflow (sqlite + S3Proxy), the same as the cloud
# MLflow App. Local mode gives no writable mount besides /opt/ml/model, so the sqlite store
# rides there (it is collected with the model); the artifacts go to S3Proxy over the host.
environment = {}
if os.environ.get("MLFLOW") == "1":
    environment = {
        "MLFLOW_TRACKING_URI": "sqlite:////opt/ml/model/mlruns.db",
        "MLFLOW_S3_ENDPOINT_URL": "http://host.docker.internal:9000",
        "MLFLOW_ARTIFACT_ROOT": "s3://mlflow-artifacts",
        "AWS_ACCESS_KEY_ID": "local",
        "AWS_SECRET_ACCESS_KEY": "localsecret",
        "AWS_DEFAULT_REGION": "us-east-1",
    }

if environment and SHARED_DB.exists():
    # seed the container's /opt/ml/model (mounted from <staging>/model) with the running
    # store so MLflow appends this run to it rather than starting a fresh database
    (staging / "model").mkdir(parents=True, exist_ok=True)
    shutil.copy(SHARED_DB, staging / "model" / "mlruns.db")

trainer = ModelTrainer(
    training_image=IMAGE,
    training_mode=Mode.LOCAL_CONTAINER,
    compute=Compute(instance_type="local", instance_count=1),
    hyperparameters={"C": "1.0", "max_iter": "1000", "monotonic": "true"},
    environment=environment,
    local_container_root=str(staging),
)
trainer.train(
    input_data_config=[
        InputData(channel_name="train", data_source=str(CHANNELS / "train")),
        InputData(channel_name="validation", data_source=str(CHANNELS / "validation")),
    ],
    wait=True,
)

# the SDK collects /opt/ml/model into compressed_artifacts/model.tar.gz; unpack it into
# runs-local/model(-challenger), where the pipeline and the serving scripts read it
MODEL_DIR.mkdir(parents=True, exist_ok=True)
with tarfile.open(staging / "compressed_artifacts" / "model.tar.gz") as tar:
    tar.extractall(MODEL_DIR)
shutil.rmtree(staging, ignore_errors=True)

# save the run back to the shared store (with this run appended) for the next model and the ui
if environment and (MODEL_DIR / "mlruns.db").exists():
    SHARED_DB.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(MODEL_DIR / "mlruns.db", SHARED_DB)
print("model ->", MODEL_DIR, sorted(p.name for p in MODEL_DIR.iterdir()))
