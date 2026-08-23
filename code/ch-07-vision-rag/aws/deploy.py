# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["sagemaker>=3,<4", "boto3", "botocore[crt]", "torch",
#                 "facenet-pytorch", "numpy<2", "scipy<1.14", "pillow",
#                 "psycopg[binary]", "psycopg-pool", "cloudpickle", "pip"]
# ///
"""Deploy the KYC face service as a SageMaker asynchronous endpoint on a GPU instance.

The same ModelBuilder and the same FaceEmbeddingSpec the laptop serves with
Mode.IN_PROCESS, with Mode.SAGEMAKER_ENDPOINT and an AsyncInferenceConfig instead.

Asynchronous rather than real-time because it is the shape that scales to zero: the
endpoint keeps no instance running between applicants, queues what arrives while it is
at zero, and writes each answer to S3. Onboarding tolerates the wake-up; a payment
would not.

Env: IMAGE_URI, SAGEMAKER_ROLE_ARN, IMAGES_BUCKET, PGHOST, DB_SECRET_ARN (the Makefile
     fills them from the stack outputs); ENDPOINT_NAME, INSTANCE_TYPE, SNS_TOPIC_ARN.

Usage:
  uv run aws/deploy.py            # create model + async endpoint, wait for InService
  uv run aws/deploy.py --delete   # remove endpoint, config, and model
"""

import argparse
import contextlib
import os
import time
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError

REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
NAME = os.environ.get("ENDPOINT_NAME", "ch07-kyc-face")
INSTANCE_TYPE = os.environ.get("INSTANCE_TYPE", "ml.g4dn.xlarge")
MAX_INSTANCES = int(os.environ.get("MAX_INSTANCES", "1"))

# ModelBuilder names the models it creates model-<hash>, not after the endpoint, so a
# teardown that only looks for ENDPOINT_NAME leaves one behind per deploy. They are
# identified instead by the serving image they point at.
SERVING_REPO = os.environ.get("SERVING_REPO", "ch07-kyc-serving")


def _status(sm, name: str) -> str | None:
    """Return the endpoint's status, or None if it does not exist."""
    try:
        return sm.describe_endpoint(EndpointName=name)["EndpointStatus"]
    except ClientError:
        return None


def _await_settled(sm, name: str, timeout: int = 2400) -> str:
    """Poll until the endpoint stops changing, tolerating transient client errors.

    A single long-lived waiter is fragile here: one TLS or credential blip aborts a
    deploy that is otherwise succeeding, and the endpoint keeps creating regardless.
    """
    deadline = time.time() + timeout
    last = "Creating"
    while time.time() < deadline:
        try:
            last = _status(sm, name) or last
        except (BotoCoreError, ClientError) as exc:  # a blip, not a verdict
            print(f"  (transient: {type(exc).__name__}, retrying)")
        if last not in ("Creating", "Updating", "SystemUpdating", "Deleting"):
            return last
        time.sleep(20)
    return last


def _orphans(sm) -> list[str]:
    """Return the generated models that point at this chapter's serving image."""
    found = []
    for page in sm.get_paginator("list_models").paginate():
        for model in page["Models"]:
            name = model["ModelName"]
            if not name.startswith("model-"):
                continue
            try:
                container = sm.describe_model(ModelName=name).get(
                    "PrimaryContainer", {}
                )
            except ClientError:
                continue
            if SERVING_REPO in container.get("Image", ""):
                found.append(name)
    return found


def _clean(sm) -> None:
    """Remove any previous endpoint, config, and model so a re-run starts fresh."""
    if _status(sm, NAME) in ("Creating", "Updating", "SystemUpdating"):
        print(f"{NAME} is still changing; AWS refuses a delete until it settles")
        _await_settled(sm, NAME)
    deletes = [
        lambda: sm.delete_endpoint(EndpointName=NAME),
        lambda: sm.delete_endpoint_config(EndpointConfigName=NAME),
        lambda: sm.delete_model(ModelName=NAME),
    ]
    orphans = _orphans(sm)
    deletes += [lambda name=n: sm.delete_model(ModelName=name) for n in orphans]
    for delete in deletes:
        with contextlib.suppress(sm.exceptions.ClientError):
            delete()
    if orphans:
        print(f"removed {len(orphans)} generated model(s)")


def _variant_name(sm, endpoint: str) -> str:
    """Return the variant's actual name; ModelBuilder calls it AllTraffic, not main."""
    config = sm.describe_endpoint(EndpointName=endpoint)["EndpointConfigName"]
    variants = sm.describe_endpoint_config(EndpointConfigName=config)[
        "ProductionVariants"
    ]
    return variants[0]["VariantName"]


def _scale_to_zero(endpoint: str) -> None:
    """Register the variant with Application Auto Scaling and let it reach zero.

    Only an asynchronous endpoint may hold a minimum capacity of zero; a real-time
    endpoint's floor is one instance, which is the cost difference between them.
    """
    aas = boto3.client("application-autoscaling", region_name=REGION)
    sm = boto3.client("sagemaker", region_name=REGION)
    resource_id = f"endpoint/{endpoint}/variant/{_variant_name(sm, endpoint)}"
    aas.register_scalable_target(
        ServiceNamespace="sagemaker",
        ResourceId=resource_id,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        MinCapacity=0,
        MaxCapacity=MAX_INSTANCES,
    )
    aas.put_scaling_policy(
        PolicyName=f"{endpoint}-backlog",
        ServiceNamespace="sagemaker",
        ResourceId=resource_id,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        PolicyType="TargetTrackingScaling",
        TargetTrackingScalingPolicyConfiguration={
            # The queue depth per instance is the signal that matters for an async
            # endpoint: it is non-zero while work is waiting, including at zero instances.
            "TargetValue": 1.0,
            "CustomizedMetricSpecification": {
                "MetricName": "ApproximateBacklogSizePerInstance",
                "Namespace": "AWS/SageMaker",
                "Dimensions": [{"Name": "EndpointName", "Value": endpoint}],
                "Statistic": "Average",
            },
            "ScaleInCooldown": 300,
            "ScaleOutCooldown": 60,
        },
    )
    # Target tracking alone does not wake an endpoint that is already at zero: it only
    # scales up once the backlog exceeds the target value, which for a queue that is
    # empty most of the day may never happen promptly. The documented remedy is a
    # second, step-scaling policy driven by HasBacklogWithoutCapacity, a metric that is
    # non-zero exactly when requests are waiting and no instance exists to take them.
    step = aas.put_scaling_policy(
        PolicyName=f"{endpoint}-wake-from-zero",
        ServiceNamespace="sagemaker",
        ResourceId=resource_id,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        PolicyType="StepScaling",
        StepScalingPolicyConfiguration={
            "AdjustmentType": "ChangeInCapacity",
            "MetricAggregationType": "Average",
            "Cooldown": 300,
            "StepAdjustments": [
                {"MetricIntervalLowerBound": 0, "ScalingAdjustment": 1},
            ],
        },
    )
    boto3.client("cloudwatch", region_name=REGION).put_metric_alarm(
        AlarmName=f"{endpoint}-has-backlog-without-capacity",
        MetricName="HasBacklogWithoutCapacity",
        Namespace="AWS/SageMaker",
        Statistic="Average",
        Dimensions=[{"Name": "EndpointName", "Value": endpoint}],
        Period=60,
        EvaluationPeriods=2,
        DatapointsToAlarm=2,
        Threshold=1,
        ComparisonOperator="GreaterThanOrEqualToThreshold",
        TreatMissingData="missing",
        AlarmActions=[step["PolicyARN"]],
    )
    print(f"autoscaling registered: {resource_id} min=0 max={MAX_INSTANCES}")
    print("wake-from-zero alarm armed on HasBacklogWithoutCapacity")


def deploy() -> None:
    """Build the model and put it behind an asynchronous endpoint that scales to zero."""
    import sys

    # ModelBuilder pickles the spec and re-imports it in a subprocess that runs from
    # its own staging directory, so src/ has to be on the path absolutely. A relative
    # PYTHONPATH resolves against that directory instead and the subprocess fails with
    # nothing more useful than a non-zero exit status.
    src = str(Path(__file__).resolve().parents[1] / "src")
    sys.path.insert(0, src)
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [src, *filter(None, [os.environ.get("PYTHONPATH")])]
    )

    from face_spec import FaceEmbeddingSpec
    from sagemaker.core.helper.session_helper import Session
    from sagemaker.core.inference_config import AsyncInferenceConfig
    from sagemaker.serve.builder.schema_builder import SchemaBuilder
    from sagemaker.serve.mode.function_pointers import Mode
    from sagemaker.serve.model_builder import ModelBuilder
    from sagemaker.serve.utils.types import ModelServer

    role = os.environ["SAGEMAKER_ROLE_ARN"]
    image = os.environ["IMAGE_URI"]
    bucket = os.environ["IMAGES_BUCKET"]
    _clean(boto3.client("sagemaker", region_name=REGION))

    # ModelBuilder stages through a session bucket even when there is nothing to
    # package. Left to itself it reaches for sagemaker-{region}-{account}, which this
    # chapter's least-privilege role cannot see (HeadBucket 403), so point it at the
    # stack's own bucket, the way Chapter 5 pins default_bucket on its pipeline session.
    session = Session(default_bucket=bucket, default_bucket_prefix="modelbuilder")

    sample = {"op": "match", "key": "probes/x.jpg", "claim": "subject_000"}
    builder = ModelBuilder(
        sagemaker_session=session,
        # our own serving image: the weights are baked in, so there is no model
        # artifact to package and nothing for ModelBuilder to build.
        image_uri=image,
        # our image is not a first-party SageMaker one, so the server has to be named:
        # MMS is the generic serve / ping / invocations runner, the contract serve.py
        # implements. Chapter 4 names the same one for its local container endpoint.
        model_server=ModelServer.MMS,
        # The same spec the laptop serves with Mode.IN_PROCESS. ModelBuilder pickles
        # it and detects its requirements, which needs pip present in the environment
        # running this script: a uv environment has none unless it is declared.
        inference_spec=FaceEmbeddingSpec(),
        # Dependency auto-detection is off. It pickles the spec and re-imports it in a
        # subprocess the SDK spawns with a hardcoded environment (env={"SETUPTOOLS_USE_
        # DISTUTILS": "stdlib"}) from inside site-packages, so nothing on PYTHONPATH is
        # importable there and it fails with a bare non-zero exit. Nothing is lost: the
        # serving image already carries every dependency, pinned in its Dockerfile.
        dependencies={"auto": False},
        schema_builder=SchemaBuilder(
            sample_input=sample, sample_output={"matched": True, "device": "cuda"}
        ),
        role_arn=role,
        # The instance type has to be set here, not only on deploy(). ModelBuilder
        # resolves a default in __post_init__ when the constructor has none, and that
        # default (ml.m5.large) wins: the endpoint comes up on a CPU box and the
        # container reports "no NVIDIA driver" while the deploy call looks correct.
        instance_type=INSTANCE_TYPE,
        mode=Mode.SAGEMAKER_ENDPOINT,
        env_vars={
            "PGHOST": os.environ["PGHOST"],
            "PGPORT": os.environ.get("PGPORT", "5432"),
            "PGUSER": os.environ.get("PGUSER", "postgres"),
            "PGDATABASE": os.environ.get("PGDATABASE", "postgres"),
            "DB_IAM_AUTH": "1",
            "IMAGES_S3_BUCKET": bucket,
            "AWS_REGION": REGION,
            "REQUIRE_GPU": os.environ.get("REQUIRE_GPU", "1"),
        },
    )
    builder.build()

    topic = os.environ.get("SNS_TOPIC_ARN")
    notification = {"SuccessTopic": topic, "ErrorTopic": topic} if topic else None
    print(f"creating {NAME} on {INSTANCE_TYPE} ...")
    builder.deploy(
        endpoint_name=NAME,
        initial_instance_count=1,
        instance_type=INSTANCE_TYPE,
        inference_config=AsyncInferenceConfig(
            output_path=f"s3://{bucket}/async-out/",
            # Without a failure path a failed inference writes nothing anywhere, and
            # the caller cannot tell "still running" from "died".
            failure_path=f"s3://{bucket}/async-fail/",
            max_concurrent_invocations_per_instance=4,
            notification_config=notification,
        ),
        wait=True,
    )

    sm = boto3.client("sagemaker", region_name=REGION)
    status = _await_settled(sm, NAME)
    if status != "InService":
        reason = sm.describe_endpoint(EndpointName=NAME).get("FailureReason", "")
        raise SystemExit(f"endpoint {NAME} is {status}: {reason}")
    print("InService:", NAME)

    _scale_to_zero(NAME)
    print("the endpoint will drop to zero instances once the queue stays empty")


def delete() -> None:
    """Tear the endpoint down; it bills for every instance-hour it stays up."""
    sm = boto3.client("sagemaker", region_name=REGION)
    aas = boto3.client("application-autoscaling", region_name=REGION)
    with contextlib.suppress(aas.exceptions.ObjectNotFoundException, ClientError):
        # Deregistering the target drops its policies with it, but the alarm that fires
        # one of them is a CloudWatch resource and outlives the endpoint on its own.
        aas.deregister_scalable_target(
            ServiceNamespace="sagemaker",
            ResourceId=f"endpoint/{NAME}/variant/{_variant_name(sm, NAME)}",
            ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        )
    boto3.client("cloudwatch", region_name=REGION).delete_alarms(
        AlarmNames=[f"{NAME}-has-backlog-without-capacity"]
    )
    _clean(sm)
    print("deleted", NAME)


def main() -> None:
    """Deploy or delete the asynchronous endpoint."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--delete", action="store_true")
    delete() if ap.parse_args().delete else deploy()


if __name__ == "__main__":
    main()
