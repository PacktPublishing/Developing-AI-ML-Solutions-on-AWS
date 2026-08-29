"""Run a pipeline's LambdaStep under SageMaker local mode.

Local mode executes Processing, Training, Transform, CreateModel, Fail and Condition
steps and rejects everything else, so a pipeline that ends in a LambdaStep fails
validation with "Step type Lambda is not supported in local mode." That is a gap in the
local executor rather than in Lambda, and this module closes it the way the chapter
closes the EventBridge and SNS gaps: a small shim the project owns.

Importing it teaches the local executor two things. First, that a Lambda step is
allowed. Second, how to run one: the step's inputs become the event, and the function
is invoked either against a local Lambda endpoint (LAMBDA_ENDPOINT, which is what
`sam local start-lambda` serves) or, with no endpoint, by importing the handler and
calling it in this process. Either way the same handler runs on the same event as on
AWS, and the step's outputs are populated the same way.

Import it only in local mode; on AWS the real executor runs the real step.
"""

import importlib
import json
import os

import boto3
from sagemaker.mlops.local import pipeline as _local_pipeline
from sagemaker.mlops.local import pipeline_entities as _entities
from sagemaker.mlops.workflow.steps import StepTypeEnum

LAMBDA_ENDPOINT = os.environ.get("LAMBDA_ENDPOINT", "")
# function name -> "module.function", so a pipeline with more than one Lambda step
# knows which handler each step should run. LAMBDA_HANDLERS is JSON, e.g.
# {"ch05-decide": "decide.handler", "ch05-apply-limits": "handler.handler"}
LAMBDA_HANDLERS = json.loads(os.environ.get("LAMBDA_HANDLERS", "{}"))
LAMBDA_HANDLER = os.environ.get("LAMBDA_HANDLER", "handler.handler")


def _handler_for(function_arn: str) -> str:
    """Pick the handler for this function, by the name at the end of its ARN."""
    return LAMBDA_HANDLERS.get(function_arn.rsplit(":", 1)[-1], LAMBDA_HANDLER)


def _invoke(event: dict, function_arn: str) -> dict:
    """Run the function on this event, through a local endpoint or in process."""
    # the step containers reach the warehouse by its docker host name; the handler runs
    # here, on the host, so it needs the published port instead
    if os.environ.get("LAMBDA_WAREHOUSE_DSN"):
        os.environ["WAREHOUSE_DSN"] = os.environ["LAMBDA_WAREHOUSE_DSN"]
    if LAMBDA_ENDPOINT:
        client = boto3.client("lambda", endpoint_url=LAMBDA_ENDPOINT)
        response = client.invoke(
            FunctionName=function_arn.rsplit(":", 1)[-1],
            Payload=json.dumps(event).encode(),
        )
        return json.loads(response["Payload"].read())
    module_name, _, func_name = _handler_for(function_arn).rpartition(".")
    module = importlib.import_module(module_name)
    return getattr(module, func_name)(event, None)


class _LocalLambdaStepExecutor(_local_pipeline._StepExecutor):
    """Execute a LambdaStep locally: inputs in, the handler's dict out."""

    def execute(self):
        """Invoke the function and return its response as the step's outputs."""
        # the step's inputs are the event; evaluate_step_arguments resolves any
        # references to earlier steps' outputs the same way it does for the others
        resolved = self.pipline_executor.evaluate_step_arguments(self.step)
        event = {
            k: v
            for k, v in resolved.items()
            if k not in ("FunctionArn", "OutputParameters")
        }
        return _invoke(event, resolved["FunctionArn"])


def install() -> None:
    """Patch the local executor to accept and run Lambda steps."""
    # 1. let validation through
    original_init = _entities._LocalPipelineExecution._initialize_step_execution

    def _initialize_step_execution(self, steps):
        """Treat a Lambda step as supported, then defer to the original check."""
        lambdas = [
            s for s in steps if getattr(s, "step_type", None) is StepTypeEnum.LAMBDA
        ]
        original_init(self, [s for s in steps if s not in lambdas])
        for step in lambdas:
            self.step_execution[step.name] = _entities._LocalPipelineExecutionStep(
                step.name, step.step_type, step.description, step.display_name
            )

    _entities._LocalPipelineExecution._initialize_step_execution = (
        _initialize_step_execution
    )

    # 2. give the factory something to run
    original_get = _local_pipeline._StepExecutorFactory.get

    def get(self, step):
        """Return the Lambda executor for Lambda steps, else the SDK's own."""
        if getattr(step, "step_type", None) is StepTypeEnum.LAMBDA:
            return _LocalLambdaStepExecutor(self.pipeline_executor, step)
        return original_get(self, step)

    _local_pipeline._StepExecutorFactory.get = get
