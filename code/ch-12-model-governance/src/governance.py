# /// script
# dependencies = ["boto3", "botocore[crt]"]
# ///
"""Generate model cards, registry metadata and lineage records.

Read governance/models.json and write local JSON records. With --upload,
create or update SageMaker model cards, register new model package versions,
and create dataset-to-model lineage associations.

Usage:
  uv run src/governance.py --models governance/models.json --out outputs/governance
  uv run src/governance.py --upload --region us-east-1
"""

import argparse
import json
import os
from pathlib import Path

import boto3


def model_card(name: str, meta: dict) -> dict:
    """Render a SageMaker model-card document (the create_model_card Content schema)."""
    metrics = meta["metrics"]
    features = meta["features"]
    feature_note = (
        f"Numeric features: {', '.join(features['numeric'])}. "
        f"Categorical features: {', '.join(features['categorical'])}."
    )
    return {
        "model_overview": {
            "model_name": name,
            "model_description": meta["description"],
            "algorithm_type": meta["algorithm"],
            "problem_type": meta["problem_type"],
            "model_owner": meta["owner"],
        },
        "intended_uses": {
            "purpose_of_model": meta["intended_use"],
            "intended_uses": meta["intended_use"],
            "factors_affecting_model_efficiency": "Trained on the chapter's synthetic "
            "credit-bureau population; efficiency degrades on populations that drift "
            "from it (see the monitoring chapter).",
            "risk_rating": meta["risk_rating"],
            "explanations_for_risk_rating": meta["risk_rationale"],
        },
        "training_details": {
            "objective_function": "Probability of default (binary log loss)",
            "training_observations": feature_note,
            "training_job_details": {
                "training_datasets": [meta["dataset"]],
            },
        },
        "evaluation_details": [
            {
                "name": meta["metrics_source"],
                "metric_groups": [
                    {
                        "name": "discrimination",
                        "metric_data": [
                            {"name": k, "type": "number", "value": v}
                            for k, v in metrics.items()
                        ],
                    }
                ],
            }
        ],
        "additional_information": {
            "ethical_considerations": "Credit decisions are high-impact; the model is "
            "one input to a human-reviewed decision, not an automated adverse action.",
            "caveats_and_recommendations": meta["out_of_scope"],
        },
    }


def registry(models: dict) -> dict:
    """Render a registry: one model package group per model, with its stage."""
    return {
        "model_package_groups": [
            {
                "model_package_group_name": name,
                "stage": meta["stage"],
                "latest_version": 1,
                "model_card": f"{name}-card.json",
                "lineage": f"{name}-lineage.json",
                "metrics": meta["metrics"],
            }
            for name, meta in models.items()
        ]
    }


def lineage(name: str, meta: dict) -> dict:
    """Generate declared dataset, training, model and deployment relationships."""
    return {
        "model": name,
        "associations": [
            {"source": meta["dataset"], "type": "DataSet", "action": "ContributedTo"},
            {"source": f"train-{name}", "type": "TrainingJob", "action": "Produced"},
            {"source": f"{name}:1", "type": "Model", "action": "DeployedTo"},
            {
                "source": "credit-rollout-gateway",
                "type": "Endpoint",
                "action": "Serves",
            },
        ],
    }


# -------------------------------------------------------------------------------
# The AWS round: the same records uploaded to SageMaker
# -------------------------------------------------------------------------------
def _already_exists(err) -> bool:
    """Report whether a create failed only because the resource already exists."""
    code = err.response["Error"]["Code"]
    msg = err.response["Error"]["Message"].lower()
    return code in ("ValidationException", "ConflictException", "ResourceInUse") and (
        "exist" in msg or "already" in msg or "in use" in msg
    )


def upload_card(sm, name: str, card: dict) -> None:
    """Create the model's SageMaker Model Card, or update it if it already exists."""
    from botocore.exceptions import ClientError

    content = json.dumps(card)
    try:
        sm.create_model_card(
            ModelCardName=name, Content=content, ModelCardStatus="Draft"
        )
        print(f"  model card created: {name}")
    except ClientError as err:
        if not _already_exists(err):
            raise
        sm.update_model_card(ModelCardName=name, Content=content)
        print(f"  model card updated: {name}")


def upload_registry(
    sm,
    name: str,
    meta: dict,
    account: str,
    region: str,
    bucket: str,
    image_tag: str,
) -> str:
    """Create a model package version and return its ARN.

    Set champion packages to Approved and other packages to PendingManualApproval.
    Use the image and artifact locations derived from the model metadata.
    """
    from botocore.exceptions import ClientError

    try:
        sm.create_model_package_group(
            ModelPackageGroupName=name,
            ModelPackageGroupDescription=meta["description"][:1024],
        )
        print(f"  registry group created: {name}")
    except ClientError as err:
        if not _already_exists(err):
            raise

    # A registry records what was deployed, so it records a tag that cannot
    # move. With :latest a package version points at whatever that tag means
    # whenever someone reads it, which is the opposite of the point.
    image = f"{account}.dkr.ecr.{region}.amazonaws.com/{meta['ecr_repo']}:{image_tag}"
    model_data = f"s3://{bucket}/{meta['model_artifact']}"
    # The URI names a key, and a key can be overwritten. The ETag names the
    # bytes that were registered, so a reader can tell later whether what the
    # endpoint loads is still what was approved here.
    etag = (
        boto3.client("s3", region_name=region)
        .head_object(Bucket=bucket, Key=meta["model_artifact"])["ETag"]
        .strip('"')
    )
    approval = "Approved" if meta["stage"] == "champion" else "PendingManualApproval"
    arn = sm.create_model_package(
        ModelPackageGroupName=name,
        ModelPackageDescription=f"{meta['algorithm']} ({meta['stage']})",
        InferenceSpecification={
            "Containers": [{"Image": image, "ModelDataUrl": model_data}],
            "SupportedContentTypes": ["application/json"],
            "SupportedResponseMIMETypes": ["application/json"],
        },
        ModelApprovalStatus=approval,
        CustomerMetadataProperties={
            "stage": meta["stage"],
            # The URI names a key and a key can be overwritten; the ETag names
            # the bytes this version was registered against.
            "model_artifact_etag": etag,
            **{k: str(v) for k, v in meta["metrics"].items()},
        },
    )["ModelPackageArn"]
    print(f"  registered {name} -> {approval} (artifact etag {etag})")
    return arn


def upload_lineage(
    sm,
    name: str,
    model_package_arn: str,
    account: str,
    region: str,
    bucket: str,
) -> None:
    """Record dataset -> model lineage as two artifacts and an association."""
    from botocore.exceptions import ClientError

    def artifact(source_uri: str, artifact_type: str, artifact_name: str) -> str:
        try:
            return sm.create_artifact(
                ArtifactName=artifact_name,
                Source={"SourceUri": source_uri},
                ArtifactType=artifact_type,
            )["ArtifactArn"]
        except ClientError as err:
            if not _already_exists(err):
                raise
            summaries = sm.list_artifacts(SourceUri=source_uri)["ArtifactSummaries"]
            return summaries[0]["ArtifactArn"]

    dataset_arn = artifact(
        f"s3://{bucket}/ch12/datasets/{name}",
        "DataSet",
        f"{name}-dataset",
    )
    model_arn = artifact(model_package_arn, "Model", f"{name}-model")
    try:
        sm.add_association(
            SourceArn=dataset_arn,
            DestinationArn=model_arn,
            AssociationType="ContributedTo",
        )
    except ClientError as err:
        if not _already_exists(err):
            raise
    print(f"  lineage: dataset ContributedTo model ({name})")


def upload(models: dict, region: str, bucket: str, image_tag: str) -> None:
    """Upload every model's card, registry entry, and lineage to SageMaker."""
    import boto3

    sm = boto3.client("sagemaker", region_name=region)
    account = boto3.client("sts", region_name=region).get_caller_identity()["Account"]
    for name, meta in models.items():
        print(f"{name}:")
        upload_card(sm, name, model_card(name, meta))
        arn = upload_registry(sm, name, meta, account, region, bucket, image_tag)
        upload_lineage(sm, name, arn, account, region, bucket)
    print(f"uploaded {len(models)} models to SageMaker governance")


def main() -> None:
    """Write a card, a registry entry, and a lineage record for every model."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", type=Path, default=Path("governance/models.json"))
    p.add_argument("--out", type=Path, default=Path("outputs/governance"))
    p.add_argument(
        "--upload",
        action="store_true",
        help="also upload to SageMaker Model Cards, Registry, and Lineage",
    )
    p.add_argument("--region", default="us-east-1")
    # The artifacts were read from sagemaker-<region>-<account>, the account's
    # own default bucket: no chapter owns it, nothing here puts the artifacts
    # in it, and no least-privilege policy can scope to it.
    p.add_argument(
        "--bucket",
        default=os.environ.get("ARTIFACT_BUCKET", ""),
        help="the bucket holding the model artifacts, from the chapter stack",
    )
    p.add_argument(
        "--image-tag",
        default=os.environ.get("IMAGE_TAG", ""),
        help="the tag the serving images were pushed under",
    )
    a = p.parse_args()
    if a.upload and not (a.bucket and a.image_tag):
        raise SystemExit(
            "--upload needs --bucket and --image-tag, or ARTIFACT_BUCKET and "
            "IMAGE_TAG. A registry entry naming neither records nothing "
            "anyone can go back to."
        )

    models = json.loads(a.models.read_text())
    a.out.mkdir(parents=True, exist_ok=True)
    for name, meta in models.items():
        (a.out / f"{name}-card.json").write_text(
            json.dumps(model_card(name, meta), indent=2)
        )
        (a.out / f"{name}-lineage.json").write_text(
            json.dumps(lineage(name, meta), indent=2)
        )
    (a.out / "registry.json").write_text(json.dumps(registry(models), indent=2))
    print(
        f"wrote model cards, registry, and lineage for {len(models)} models -> {a.out}"
    )
    if a.upload:
        upload(models, a.region, a.bucket, a.image_tag)


if __name__ == "__main__":
    main()
