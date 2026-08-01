# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = ["sagemaker>=3,<4", "sagemaker-mlops", "botocore[crt]", "boto3"]
# ///
"""The batch limit run as a SageMaker Pipeline — one script, local AND AWS.

Four ProcessingSteps (shortlist -> score -> decide -> apply), data flowing
between them by S3 property reference. Only the session changes between
worlds: LocalPipelineSession (Docker, no quota) vs PipelineSession
(SageMaker jobs).

Env: STEP_IMAGE (local tag or ECR URI), SAGEMAKER_ROLE_ARN, BATCH_BUCKET,
     WAREHOUSE_DSN (as reachable from inside the step containers),
     PIPELINE_MODE (local|aws), INSTANCE_TYPE (aws only).

Run local:  docker build -t ch05-step:local pipeline/step_image/
  PIPELINE_MODE=local STEP_IMAGE=ch05-step:local BATCH_BUCKET=portfolio \
  WAREHOUSE_DSN=postgresql://analyst:analyst@host.docker.internal:5439/portfolio \
  SAGEMAKER_ROLE_ARN=arn:aws:iam::111111111111:role/local-dummy \
  uv run pipeline/pipeline.py
"""

import json
import os

from sagemaker.core.processing import (
    PipelineSession,
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor,
)
from sagemaker.core.shapes import ProcessingS3Input, ProcessingS3Output
from sagemaker.mlops.local.local_pipeline_session import (
    LocalPipelineSession as _MlopsLocalPipelineSession,
)
from sagemaker.mlops.workflow.pipeline import Pipeline
from sagemaker.mlops.workflow.steps import ProcessingStep

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(HERE, "scripts")
MODE = os.environ.get("PIPELINE_MODE", "local")
IMAGE = os.environ["STEP_IMAGE"]
ROLE = os.environ["SAGEMAKER_ROLE_ARN"]
BUCKET = os.environ["BATCH_BUCKET"]
WAREHOUSE_DSN = os.environ["WAREHOUSE_DSN"]
IO_PREFIX = os.environ.get("IO_PREFIX", "batch/pipeline-io")

with open(os.path.join(HERE, "..", "data", "feature_spec.json")) as f:
    FEATURES = ",".join(json.load(f)["features"])


class LocalPipelineSession(_MlopsLocalPipelineSession, PipelineSession):
    """The local pipeline session v3 is missing: both halves in one class.

    v3 ships two LocalPipelineSessions and each has half of what a local
    pipeline needs; inheriting from both keeps the mlops implementations and
    makes isinstance(sess, PipelineSession) true, so steps compose into a
    DAG instead of executing on definition (same workaround as chapter 2).
    """


if MODE == "local":
    sess = LocalPipelineSession()
    instance_type = "local"
else:
    sess = PipelineSession()
    instance_type = os.environ.get("INSTANCE_TYPE", "ml.m5.large")


def processor(name):
    """Build a ScriptProcessor on the shared step image."""
    return ScriptProcessor(
        image_uri=IMAGE,
        command=["python3"],
        instance_count=1,
        instance_type=instance_type,
        base_job_name=name,
        role=ROLE,
        sagemaker_session=sess,
        env={"WAREHOUSE_DSN": WAREHOUSE_DSN, "FEATURES": FEATURES},
    )


def step_output(step_name, output_name, local_path):
    """One ProcessingOutput with a deterministic S3 home."""
    return ProcessingOutput(
        output_name=output_name,
        s3_output=ProcessingS3Output(
            s3_uri=f"s3://{BUCKET}/{IO_PREFIX}/{step_name}/{output_name}",
            local_path=local_path,
            s3_upload_mode="EndOfJob",
        ),
    )


def step_input(input_name, s3_uri, local_path):
    """One ProcessingInput from a previous step's output."""
    return ProcessingInput(
        input_name=input_name,
        s3_input=ProcessingS3Input(
            s3_uri=s3_uri,
            local_path=local_path,
            s3_data_type="S3Prefix",
            s3_input_mode="File",
        ),
    )


def out_uri(step, name):
    """Return the S3 URI of a step's named output (a lazy pipeline variable)."""
    return step.properties.ProcessingOutputConfig.Outputs[name].S3Output.S3Uri


shortlist = ProcessingStep(
    name="shortlist",
    step_args=processor("shortlist").run(
        code=os.path.join(SCRIPTS, "shortlist.py"),
        outputs=[step_output("shortlist", "eligible", "/opt/ml/processing/output")],
    ),
)

score = ProcessingStep(
    name="score",
    step_args=processor("score").run(
        code=os.path.join(SCRIPTS, "score.py"),
        inputs=[
            step_input(
                "eligible", out_uri(shortlist, "eligible"), "/opt/ml/processing/input"
            )
        ],
        outputs=[
            step_output("score", "scores", "/opt/ml/processing/scores"),
            step_output("score", "model", "/opt/ml/processing/model"),
        ],
    ),
)

decide = ProcessingStep(
    name="decide",
    step_args=processor("decide").run(
        code=os.path.join(SCRIPTS, "decide.py"),
        inputs=[
            step_input("scores", out_uri(score, "scores"), "/opt/ml/processing/input")
        ],
        outputs=[step_output("decide", "decisions", "/opt/ml/processing/output")],
    ),
)

apply_limits = ProcessingStep(
    name="apply-limits",
    step_args=processor("apply-limits").run(
        code=os.path.join(SCRIPTS, "apply.py"),
        inputs=[
            step_input(
                "decisions", out_uri(decide, "decisions"), "/opt/ml/processing/input"
            )
        ],
        outputs=[step_output("apply-limits", "summary", "/opt/ml/processing/output")],
    ),
)

pipeline = Pipeline(
    name="batch-credit-limits",
    steps=[shortlist, score, decide, apply_limits],
    sagemaker_session=sess,
)

if __name__ == "__main__":
    # v3 moved the pipeline calls from the local client to the local session:
    # register through the session locally, upsert only on AWS (same as ch-02).
    if MODE == "local":
        sess.create_pipeline(pipeline, "batch limit run, local execution")
        execution = sess.start_pipeline_execution(PipelineName=pipeline.name)
    else:
        pipeline.upsert(role_arn=ROLE)
        execution = pipeline.start()
        execution.wait()
    print(f"pipeline started ({MODE})")
