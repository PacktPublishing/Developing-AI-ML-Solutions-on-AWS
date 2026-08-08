# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["sagemaker>=3,<4", "sagemaker-mlops", "sagemaker-serve", "botocore[crt]", "boto3"]
# ///
"""Champion/challenger as a SageMaker Pipeline: one script, local AND AWS.

Five traditional ProcessingSteps, data flowing between them by S3 property reference:

  prepare (Processing) -> train-scorecard (Processing) ┐
                          train-challenger (Processing) ┘-> select (Processing)
                          -> export (Processing) -> scores.csv in S3

All steps are ProcessingSteps, so every step runs in the local executor and on AWS at full parity; only the session changes (LocalPipelineSession vs PipelineSession). SDK v3 throughout; traditional ProcessingSteps not @step, whose arguments serialize eagerly and cannot feed a ProcessingStep.

Env: STEP_IMAGE (local tag or ECR URI), SAGEMAKER_ROLE_ARN, SCORE_BUCKET,
     PIPELINE_MODE (local|aws), INSTANCE_TYPE (aws only), SCORE_PREFIX (optional).

Run local:  docker build -t ch02-step:local pipeline/step_image/
  PIPELINE_MODE=local STEP_IMAGE=ch02-step:local SCORE_BUCKET=<bucket> \
  SAGEMAKER_ROLE_ARN=arn:aws:iam::<acct>:role/<Role> uv run pipeline/pipeline.py
"""

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

# -------------------------------------------------------------------------------
# Environment and mode
# -------------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(HERE, "scripts")
MODE = os.environ.get("PIPELINE_MODE", "local")
IMAGE = os.environ["STEP_IMAGE"]
ROLE = os.environ["SAGEMAKER_ROLE_ARN"]
BUCKET = os.environ["SCORE_BUCKET"]
PREFIX = os.environ.get("SCORE_PREFIX", "ch02/pipeline-scores")
# v3 requires an explicit S3 URI on every ProcessingOutput, so step-to-step I/O gets a deterministic home that a re-run overwrites (reproducible by design).
IO_PREFIX = os.environ.get("IO_PREFIX", "ch02/pipeline-io")


# -------------------------------------------------------------------------------
# The hybrid local session
# -------------------------------------------------------------------------------
class LocalPipelineSession(_MlopsLocalPipelineSession, PipelineSession):
    """The local pipeline session v3 is missing: both halves in one class.

    v3 ships two LocalPipelineSessions, each with half of what's needed: sagemaker.mlops.local's has the pipeline methods but isn't a PipelineSession (so .run() executes immediately), and sagemaker.core.workflow's defers correctly but has no pipeline methods. Inheriting both keeps the mlops methods and makes isinstance(sess, PipelineSession) true, so steps compose into a DAG.
    """


if MODE == "local":
    sess = LocalPipelineSession()
    instance_type = "local"
else:
    sess = PipelineSession()
    # ml.m5.large; the two train steps run concurrently, so needs "ml.m5.large for processing job usage" quota >= 2 (fresh account = 0). Cold-verified 2026-08-03: all 5 steps Succeeded.
    instance_type = os.environ.get("INSTANCE_TYPE", "ml.m5.large")


# -------------------------------------------------------------------------------
# Step builders
# -------------------------------------------------------------------------------
def processor(name):
    """Build a ScriptProcessor on the shared image, one per step, runs `python3 <code>`."""
    return ScriptProcessor(
        image_uri=IMAGE,
        command=["python3"],
        instance_count=1,
        instance_type=instance_type,
        base_job_name=name,
        role=ROLE,
        sagemaker_session=sess,
    )


def step_output(step_name, output_name, local_path, s3_uri=None):
    """One ProcessingOutput: v3 nests the S3 details in a ProcessingS3Output."""
    return ProcessingOutput(
        output_name=output_name,
        s3_output=ProcessingS3Output(
            s3_uri=s3_uri or f"s3://{BUCKET}/{IO_PREFIX}/{step_name}/{output_name}",
            local_path=local_path,
            s3_upload_mode="EndOfJob",
        ),
    )


def step_input(input_name, s3_uri, local_path):
    """One ProcessingInput: v3 nests the S3 details in a ProcessingS3Input."""
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
    """Return the S3 URI of a step's named ProcessingOutput (a lazy pipeline variable)."""
    return step.properties.ProcessingOutputConfig.Outputs[name].S3Output.S3Uri


# -------------------------------------------------------------------------------
# Pipeline steps
# -------------------------------------------------------------------------------
# 1. prepare: synthesize + split the data
prepare = ProcessingStep(
    name="prepare",
    step_args=processor("prepare").run(
        code=os.path.join(SCRIPTS, "prepare.py"),
        outputs=[
            step_output("prepare", "train", "/opt/ml/processing/train"),
            step_output("prepare", "test", "/opt/ml/processing/test"),
        ],
    ),
)


def train_step(model):
    """Build a training ProcessingStep for one model, reading prepare's train/test outputs."""
    return ProcessingStep(
        name=f"train-{model}",
        step_args=processor(f"train-{model}").run(
            code=os.path.join(SCRIPTS, "train.py"),
            arguments=["--model", model],
            inputs=[
                step_input(
                    "train", out_uri(prepare, "train"), "/opt/ml/processing/input"
                ),
                step_input("test", out_uri(prepare, "test"), "/opt/ml/processing/test"),
            ],
            outputs=[
                step_output(f"train-{model}", "model", "/opt/ml/processing/model")
            ],
        ),
    )


train_scorecard = train_step("scorecard")
train_challenger = train_step("challenger")

# 3. select: compare the two models' metrics, pick the winner
select = ProcessingStep(
    name="select",
    step_args=processor("select").run(
        code=os.path.join(SCRIPTS, "select.py"),
        inputs=[
            step_input(
                "scorecard",
                out_uri(train_scorecard, "model"),
                "/opt/ml/processing/scorecard",
            ),
            step_input(
                "challenger",
                out_uri(train_challenger, "model"),
                "/opt/ml/processing/challenger",
            ),
        ],
        outputs=[step_output("select", "winner", "/opt/ml/processing/winner")],
    ),
)

# 4. export: write the winner's scores to S3 (the requested SageMaker Processing step)
export = ProcessingStep(
    name="export",
    step_args=processor("export").run(
        code=os.path.join(SCRIPTS, "export.py"),
        inputs=[
            step_input("winner", out_uri(select, "winner"), "/opt/ml/processing/input")
        ],
        outputs=[
            step_output(
                "export",
                "scores",
                "/opt/ml/processing/output",
                s3_uri=f"s3://{BUCKET}/{PREFIX}",
            )
        ],
    ),
)

# -------------------------------------------------------------------------------
# The DAG
# -------------------------------------------------------------------------------
pipeline = Pipeline(
    name="credit-champion-challenger",
    steps=[prepare, train_scorecard, train_challenger, select, export],
    sagemaker_session=sess,
)

# -------------------------------------------------------------------------------
# Entrypoint
# -------------------------------------------------------------------------------
if __name__ == "__main__":
    # v3 moved pipeline calls to the local session: upsert()/start reach for sagemaker_client, which local mode lacks, so register through the session locally and use upsert only on AWS.
    if MODE == "local":
        # Pipeline.upsert/start route through sagemaker_client, which local mode
        # does not implement; the local session carries these methods itself.
        sess.create_pipeline(pipeline, "champion/challenger, local execution")
        execution = sess.start_pipeline_execution(PipelineName=pipeline.name)
    else:
        pipeline.upsert(role_arn=ROLE)
        execution = pipeline.start()
        execution.wait()
    print(f"pipeline started ({MODE})")
    print(f"scores -> s3://{BUCKET}/{PREFIX}/scores.csv")
