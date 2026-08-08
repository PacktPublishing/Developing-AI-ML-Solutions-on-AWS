# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["sagemaker>=3,<4", "botocore[crt]", "boto3"]
# ///
"""Deploy one model's bring-your-own serving container, local or cloud.

MODE=local runs the image as a SageMaker local endpoint (Mode.LOCAL_CONTAINER); MODE=cloud puts it behind a serverless endpoint. Same image either way.

Usage:
  MODE=local MODEL_IMAGE=<ecr-uri> SAGEMAKER_ROLE_ARN=<role> uv run src/deploy.py
  MODE=cloud MODEL_IMAGE=<ecr-uri> SAGEMAKER_ROLE_ARN=<role> \
    ENDPOINT_NAME=ch04-scorecard uv run src/deploy.py

Needs AWS credentials, a SageMaker execution role, and MODEL_IMAGE as an ECR uri (linux/amd64, Docker v2 manifest).
"""

import json
import os
import time

MODE = os.environ.get("MODE", "local")
IMAGE = os.environ["MODEL_IMAGE"]
ROLE = os.environ["SAGEMAKER_ROLE_ARN"]
ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "ch04-scorecard")

# one application's features and the shape of the answer
sample = {
    "age": 40,
    "monthly_income": 6000.0,
    "requested_amount": 5000.0,
    "dti": 0.10,
    "utilization": 0.10,
    "days_past_due": 0,
}


def deploy_local() -> None:
    """Run the container locally as a SageMaker local endpoint (Mode.LOCAL_CONTAINER)."""
    from sagemaker.serve.builder.schema_builder import SchemaBuilder
    from sagemaker.serve.mode.function_pointers import Mode
    from sagemaker.serve.model_builder import ModelBuilder
    from sagemaker.serve.utils.types import ModelServer

    builder = ModelBuilder(
        image_uri=IMAGE,
        model_server=ModelServer.MMS,  # the generic serve/ping/invocations runner
        schema_builder=SchemaBuilder(sample_input=sample, sample_output={"pd": 0.09}),
        role_arn=ROLE,
        mode=Mode.LOCAL_CONTAINER,
    )
    builder.build()
    endpoint = builder.deploy_local(wait=True, container_timeout_in_seconds=600)
    resp = endpoint.invoke(body=json.dumps(sample), content_type="application/json")
    print("prediction:", resp.body.read().decode() if hasattr(resp, "body") else resp)
    endpoint.delete()


def deploy_cloud() -> None:
    """Put the same image behind a serverless SageMaker endpoint."""
    import boto3

    sm = boto3.client("sagemaker")
    sm.create_model(
        ModelName=ENDPOINT_NAME,
        PrimaryContainer={"Image": IMAGE},  # self-contained: no ModelDataUrl
        ExecutionRoleArn=ROLE,
    )
    sm.create_endpoint_config(
        EndpointConfigName=ENDPOINT_NAME,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": ENDPOINT_NAME,
                "ServerlessConfig": {"MemorySizeInMB": 2048, "MaxConcurrency": 2},
            }
        ],
    )
    sm.create_endpoint(EndpointName=ENDPOINT_NAME, EndpointConfigName=ENDPOINT_NAME)
    print("creating endpoint", ENDPOINT_NAME, "...")
    sm.get_waiter("endpoint_in_service").wait(EndpointName=ENDPOINT_NAME)

    # InService means /ping passed; the container may take a few more seconds to
    # accept traffic, so retry the first invocation past the cold-start race.
    runtime = boto3.client("sagemaker-runtime")
    for attempt in range(6):
        try:
            resp = runtime.invoke_endpoint(
                EndpointName=ENDPOINT_NAME,
                ContentType="application/json",
                Body=json.dumps(sample),
            )
            print("endpoint InService; prediction:", resp["Body"].read().decode())
            return
        except runtime.exceptions.ModelError:
            if attempt == 5:
                raise
            time.sleep(10)


if __name__ == "__main__":
    (deploy_local if MODE == "local" else deploy_cloud)()
