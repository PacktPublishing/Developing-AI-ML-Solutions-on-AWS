# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["sagemaker>=3,<4", "sagemaker-mlops", "botocore[crt]", "boto3", "pandas"]
# ///
"""ch-11 monitoring as a SageMaker Pipeline: Batch Transform, then the drift monitor.

One script, local AND AWS, the way ch-02 and ch-05 run their pipelines. The DAG is
the same in both modes:

  score-current (Batch Transform) -> monitor-drift (Processing)

The Transform step scores the current batch through the scorecard's serving container
-- the realistic way you score a production batch, and the way Model Monitor captures
predictions. The Processing step runs run_monitor.py, the chapter's one monitoring
entrypoint, handing it the /opt/ml/processing paths as arguments: it reads those live
scores, the reference and current batches, and the model, computes PSI + SHAP
feature-attribution drift, writes the Clarify-shaped artifacts, and publishes the
metrics to CloudWatch, where the stack's alarms turn them into alerts. SageMaker
Clarify is closed to new accounts; this is AWS's documented replacement -- the same
SHAP, run as a Processing step you own.

PIPELINE_MODE selects the executor. local runs the Transform and Processing containers
on this machine through the SDK's local mode (instance_type=local, local control
plane), staging inputs and outputs to real S3 exactly as ch-05 does; aws runs them as
managed jobs. The model the Transform step scores through is created through the
session's own SageMaker client -- real on AWS, the SDK's local registry in local mode
-- so the identical code drives both.

Env: PIPELINE_MODE (local|aws, default aws), IMAGE_URI, SAGEMAKER_ROLE_ARN,
     ARTIFACT_BUCKET, AWS_DEFAULT_REGION (default us-east-1), DATA_DIR (default
     ../data/generated), MODEL_DIR (default ../runs-local/model), INSTANCE_TYPE
     (aws only, default ml.m5.large).
"""

import os
import tarfile
from pathlib import Path

import boto3
import pandas as pd
from botocore.exceptions import ClientError
from sagemaker.core.processing import (
    PipelineSession,
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor,
)
from sagemaker.core.shapes import ProcessingS3Input, ProcessingS3Output
from sagemaker.core.transformer import Transformer
from sagemaker.mlops.local.local_pipeline_session import (
    LocalPipelineSession as _MlopsLocalPipelineSession,
)
from sagemaker.mlops.workflow.pipeline import Pipeline
from sagemaker.mlops.workflow.steps import ProcessingStep, TransformStep

MODE = os.environ.get("PIPELINE_MODE", "aws")
REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
IMAGE = os.environ["IMAGE_URI"]
ROLE = os.environ["SAGEMAKER_ROLE_ARN"]
BUCKET = os.environ["ARTIFACT_BUCKET"]
DATA_DIR = Path(os.environ.get("DATA_DIR", "../data/generated"))
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "../runs-local/model"))

PREFIX = "ch11/pipeline"
MODEL_NAME = "ch11-scorecard-transform"
HERE = Path(__file__).resolve().parent
MONITOR = str(HERE.parent / "src" / "run_monitor.py")

# The scorecard's feature contract (the transform input is features only, no label).
NUMERIC = [
    "age",
    "annual_income",
    "debt_to_income",
    "bureau_score",
    "credit_utilization",
    "employment_length_years",
    "loan_amount",
]
CATEGORICAL = ["home_ownership", "loan_purpose", "employment_status"]
FEATURES = NUMERIC + CATEGORICAL

s3 = boto3.client("s3", region_name=REGION)


class LocalPipelineSession(_MlopsLocalPipelineSession, PipelineSession):
    """The v3 local pipeline session is split across two classes; join both halves."""


if MODE == "local":
    # local mode runs the containers here; the control plane is the SDK's local one,
    # so instance_type is "local" and the model lives in the local SageMaker registry.
    # Set default_bucket explicitly (the S3Proxy bucket) so uploading the step code
    # never resolves sagemaker-<region>-<account> through STS, which dummy creds fail.
    sess = LocalPipelineSession(default_bucket=BUCKET)
    INSTANCE = "local"
else:
    sess = PipelineSession(
        boto_session=boto3.Session(region_name=REGION), default_bucket=BUCKET
    )
    INSTANCE = os.environ.get("INSTANCE_TYPE", "ml.m5.large")


def _put(local: str, key: str) -> str:
    """Upload one file to the pipeline prefix and return its S3 URI."""
    s3.upload_file(local, BUCKET, f"{PREFIX}/{key}")
    return f"s3://{BUCKET}/{PREFIX}/{key}"


def stage() -> dict:
    """Stage the batches, the transform input, the model artifact, and the model.tar.gz."""
    reference = pd.read_csv(DATA_DIR / "reference.csv")
    current = pd.read_csv(DATA_DIR / "current.csv")

    reference.to_csv("/tmp/ch11-reference.csv", index=False)
    current.to_csv("/tmp/ch11-current.csv", index=False)
    # transform input: headerless features only, the format serve.py expects
    current[FEATURES].to_csv("/tmp/ch11-features.csv", header=False, index=False)
    with tarfile.open("/tmp/ch11-model.tar.gz", "w:gz") as tar:
        tar.add(MODEL_DIR / "scorecard.cbm", arcname="scorecard.cbm")

    return {
        "reference": _put("/tmp/ch11-reference.csv", "reference/reference.csv"),
        "current": _put("/tmp/ch11-current.csv", "current/current.csv"),
        "features": _put("/tmp/ch11-features.csv", "features/features.csv"),
        "model_raw": _put(str(MODEL_DIR / "scorecard.cbm"), "model/scorecard.cbm"),
        "model_tar": _put("/tmp/ch11-model.tar.gz", "model.tar.gz"),
    }


def ensure_model(model_tar: str) -> None:
    """(Re)create the model the Transform step scores through, in this session's control plane.

    Real SageMaker on AWS, the SDK's local registry in local mode. Routing through
    sess.sagemaker_client (not a bare boto3 client) is what lets the same call register
    a model the local Transform executor can find.
    """
    client = sess.sagemaker_client
    try:
        client.delete_model(ModelName=MODEL_NAME)
    except (ClientError, KeyError):
        pass
    client.create_model(
        ModelName=MODEL_NAME,
        ExecutionRoleArn=ROLE,
        PrimaryContainer={
            "Image": IMAGE,
            "ModelDataUrl": model_tar,
            "Mode": "SingleModel",
            # local mode serves the container from this dict and mutates it in place
            "Environment": {},
        },
    )


def _input(name: str, s3_uri: str) -> ProcessingInput:
    """One Processing input mounted under the container's input tree."""
    return ProcessingInput(
        input_name=name,
        s3_input=ProcessingS3Input(
            s3_uri=s3_uri,
            local_path=f"/opt/ml/processing/input/{name}",
            s3_data_type="S3Prefix",
            s3_input_mode="File",
        ),
    )


def main() -> None:
    """Stage inputs, build the transform -> monitor DAG, and run it (local or AWS)."""
    staged = stage()
    ensure_model(staged["model_tar"])

    transformer = Transformer(
        model_name=MODEL_NAME,
        instance_count=1,
        instance_type=INSTANCE,
        strategy="MultiRecord",
        assemble_with="Line",
        accept="text/csv",
        output_path=f"s3://{BUCKET}/{PREFIX}/scores",
        # name the job explicitly so the SDK does not derive it by describing the model
        # against the real control plane, which the local registry model would fail
        base_transform_job_name="ch11-score-current",
        sagemaker_session=sess,
    )
    transform_step = TransformStep(
        name="score-current",
        step_args=transformer.transform(
            data=f"s3://{BUCKET}/{PREFIX}/features/",
            data_type="S3Prefix",
            content_type="text/csv",
            split_type="Line",
        ),
    )

    # On AWS the monitor publishes to real CloudWatch; locally there is no CloudWatch to
    # reach, so MONITOR_LOCAL selects the same from-source shim the chapter's make run uses.
    monitor = ScriptProcessor(
        image_uri=IMAGE,
        command=["python3"],
        instance_count=1,
        instance_type=INSTANCE,
        base_job_name="monitor-drift",
        role=ROLE,
        env={"MONITOR_LOCAL": "1"} if MODE == "local" else {},
        sagemaker_session=sess,
    )
    monitor_step = ProcessingStep(
        name="monitor-drift",
        step_args=monitor.run(
            code=MONITOR,
            arguments=[
                "--reference",
                "/opt/ml/processing/input/reference/reference.csv",
                "--current",
                "/opt/ml/processing/input/current/current.csv",
                "--model",
                "/opt/ml/processing/input/model/scorecard.cbm",
                "--scores",
                "/opt/ml/processing/input/scores",
                "--out",
                "/opt/ml/processing/output",
            ],
            inputs=[
                _input("reference", staged["reference"]),
                _input("current", staged["current"]),
                _input("model", staged["model_raw"]),
                _input("scores", f"s3://{BUCKET}/{PREFIX}/scores/"),
            ],
            outputs=[
                ProcessingOutput(
                    output_name="analysis",
                    s3_output=ProcessingS3Output(
                        s3_uri=f"s3://{BUCKET}/{PREFIX}/out",
                        local_path="/opt/ml/processing/output",
                        s3_upload_mode="EndOfJob",
                    ),
                )
            ],
        ),
        depends_on=[transform_step],
    )

    pipeline = Pipeline(
        name="ch11-monitoring",
        steps=[transform_step, monitor_step],
        sagemaker_session=sess,
    )

    # v3 moved the pipeline calls to the local session: register and run through the
    # session locally, upsert and start on AWS (the same split as ch-02 and ch-05).
    if MODE == "local":
        sess.create_pipeline(pipeline, "ch11 monitoring, local execution")
        execution = sess.start_pipeline_execution(PipelineName=pipeline.name)
        print(f"pipeline ch11-monitoring complete ({MODE})")
        return

    pipeline.upsert(role_arn=ROLE)
    execution = pipeline.start()
    print("pipeline ch11-monitoring started; waiting ...")
    execution.wait()
    print("pipeline complete")
    try:
        steps = boto3.client(
            "sagemaker", region_name=REGION
        ).list_pipeline_execution_steps(PipelineExecutionArn=execution.arn)[
            "PipelineExecutionSteps"
        ]
        for step in steps:
            print(f"  {step['StepName']:16} {step['StepStatus']}")
    except (ClientError, KeyError, AttributeError) as exc:
        print(f"  (steps summary unavailable: {exc})")


if __name__ == "__main__":
    main()
