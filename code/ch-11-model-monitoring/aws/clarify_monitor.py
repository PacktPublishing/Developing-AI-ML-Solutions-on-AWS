# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["sagemaker>=3,<4", "sagemaker-mlops", "botocore[crt]", "boto3", "pandas", "catboost"]
# ///
"""Verify the chapter's from-source SHAP feature-attribution drift against real SageMaker Clarify.

Runs two Clarify explainability jobs (KernelSHAP, global mean-absolute) through a shadow endpoint
built from the scorecard -- one on the reference batch, one on the drifted current batch -- reads
each analysis.json, and scores the NDCG between the two feature-attribution rankings. That is the
exact measure src/attribution.py computes from CatBoost TreeSHAP. If the two NDCGs agree, and agree
on the < 0.90 drift verdict, the from-source monitor is faithful to Clarify.

SageMaker Clarify is closed to new accounts, so this parity check needs a grandfathered account;
the chapter's everyday path is the Processing-step replacement in pipeline.py (make pipeline).

Env: IMAGE_URI, SAGEMAKER_ROLE_ARN, ARTIFACT_BUCKET, AWS_DEFAULT_REGION (default us-east-1),
     DATA_DIR (default ../data/generated), MODEL_DIR (default ../runs-local/model),
     INSTANCE_TYPE (default ml.m5.large), SAMPLE_ROWS (default 200), NUM_SAMPLES (default 200).

Usage:
  AWS_PROFILE=<grandfathered> make clarify
"""

import json
import os
import tarfile
from pathlib import Path

import boto3
import pandas as pd
from attribution import (
    attribution_ndcg,
    attributions,
)  # chapter's from-source (TreeSHAP)
from model import CATEGORICAL, FEATURES, TARGET, load
from sagemaker.core.clarify import (
    DataConfig,
    ModelConfig,
    SageMakerClarifyProcessor,
    SHAPConfig,
)
from sagemaker.core.helper.session_helper import Session

REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
IMAGE = os.environ["IMAGE_URI"]
ROLE = os.environ["SAGEMAKER_ROLE_ARN"]
BUCKET = os.environ["ARTIFACT_BUCKET"]
DATA_DIR = Path(os.environ.get("DATA_DIR", "../data/generated"))
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "../runs-local/model"))
INSTANCE = os.environ.get("INSTANCE_TYPE", "ml.m5.large")
SAMPLE = int(os.environ.get("SAMPLE_ROWS", "200"))
NUM_SAMPLES = int(os.environ.get("NUM_SAMPLES", "200"))
PREFIX = "ch11/clarify"
MODEL_NAME = "ch11-scorecard-clarify"

boto = boto3.Session(region_name=REGION)
sess = Session(boto_session=boto, default_bucket=BUCKET)
s3 = boto.client("s3")


def put(local: str, key: str) -> str:
    """Upload one file to the clarify prefix and return its S3 URI."""
    s3.upload_file(local, BUCKET, f"{PREFIX}/{key}")
    return f"s3://{BUCKET}/{PREFIX}/{key}"


def stage() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Sample matching rows from both batches, stage the CSVs and the model.tar.gz to S3."""
    ref = pd.read_csv(DATA_DIR / "reference.csv")
    cur = pd.read_csv(DATA_DIR / "current.csv")
    if len(ref) > SAMPLE:
        ref = ref.sample(SAMPLE, random_state=0).reset_index(drop=True)
    if len(cur) > SAMPLE:
        cur = cur.sample(SAMPLE, random_state=0).reset_index(drop=True)
    ref.to_csv("/tmp/clarify-reference.csv", index=False)
    cur.to_csv("/tmp/clarify-current.csv", index=False)
    with tarfile.open("/tmp/clarify-model.tar.gz", "w:gz") as tar:
        tar.add(MODEL_DIR / "scorecard.cbm", arcname="scorecard.cbm")
    staged = {
        "reference": put("/tmp/clarify-reference.csv", "reference.csv"),
        "current": put("/tmp/clarify-current.csv", "current.csv"),
        "model_tar": put("/tmp/clarify-model.tar.gz", "model.tar.gz"),
    }
    return ref, cur, staged


def ensure_model(model_tar: str) -> None:
    """(Re)create the SageMaker model Clarify stands its shadow endpoint up from."""
    client = sess.sagemaker_client
    try:
        client.delete_model(ModelName=MODEL_NAME)
    except Exception:
        pass
    client.create_model(
        ModelName=MODEL_NAME,
        ExecutionRoleArn=ROLE,
        PrimaryContainer={
            "Image": IMAGE,
            "ModelDataUrl": model_tar,
            "Mode": "SingleModel",
        },
    )


def baseline_row(ref: pd.DataFrame) -> list:
    """Build one KernelSHAP baseline: the mean of each numeric feature, the mode of each categorical."""
    return [
        str(ref[f].mode().iloc[0]) if f in CATEGORICAL else float(ref[f].mean())
        for f in FEATURES
    ]


def clarify_shap(name: str, data_uri: str, baseline: list) -> pd.Series:
    """Run one Clarify explainability job and return its global mean-abs SHAP per feature."""
    processor = SageMakerClarifyProcessor(
        role=ROLE,
        instance_count=1,
        instance_type=INSTANCE,
        sagemaker_session=sess,
        job_name_prefix=f"ch11-clarify-{name}",
    )
    out = f"s3://{BUCKET}/{PREFIX}/out/{name}"
    data_config = DataConfig(
        s3_data_input_path=data_uri,
        s3_output_path=out,
        label=TARGET,
        headers=list(FEATURES) + [TARGET],
        dataset_type="text/csv",
    )
    model_config = ModelConfig(
        model_name=MODEL_NAME,
        instance_count=1,
        instance_type=INSTANCE,
        accept_type="text/csv",
        content_type="text/csv",
    )
    shap_config = SHAPConfig(
        baseline=[baseline], num_samples=NUM_SAMPLES, agg_method="mean_abs"
    )
    processor.run_explainability(data_config, model_config, shap_config, model_scores=0)
    local = f"/tmp/clarify-analysis-{name}.json"
    s3.download_file(BUCKET, f"{PREFIX}/out/{name}/analysis.json", local)
    with open(local) as f:
        values = json.load(f)["explanations"]["kernel_shap"]["default"][
            "global_shap_values"
        ]
    return pd.Series({f: float(values[f]) for f in FEATURES})


def verdict(ndcg: float) -> str:
    """Return the chapter's drift call: below 0.90 (NDCG_MIN) is drift, at or above is stable."""
    return "DRIFT" if ndcg < 0.90 else "stable"


def main() -> None:
    """Run Clarify on both batches, compute both NDCGs on identical rows, and report parity."""
    ref, cur, staged = stage()
    ensure_model(staged["model_tar"])
    baseline = baseline_row(ref)

    print(f"Clarify explainability on the reference batch ({len(ref)} rows) ...")
    clarify_ref = clarify_shap("reference", staged["reference"], baseline)
    print(f"Clarify explainability on the current batch ({len(cur)} rows) ...")
    clarify_cur = clarify_shap("current", staged["current"], baseline)
    clarify_ndcg = attribution_ndcg(clarify_ref, clarify_cur)

    model = load(MODEL_DIR / "scorecard.cbm")
    fs_ref, fs_cur = attributions(model, ref), attributions(model, cur)
    fs_ndcg = attribution_ndcg(fs_ref, fs_cur)

    print(
        "\nreference feature-attribution ranking (mean-abs SHAP), Clarify vs from-source:"
    )
    for f in clarify_ref.sort_values(ascending=False).index:
        print(f"  {f:26s} clarify={clarify_ref[f]:.4f}   from-source={fs_ref[f]:.4f}")
    print("\nfeature-attribution NDCG (reference vs current):")
    print(
        f"  real Clarify (KernelSHAP): {clarify_ndcg:.4f}   -> {verdict(clarify_ndcg)}"
    )
    print(f"  from-source  (TreeSHAP):   {fs_ndcg:.4f}   -> {verdict(fs_ndcg)}")
    agree = verdict(clarify_ndcg) == verdict(fs_ndcg)
    print(
        f"  delta {abs(clarify_ndcg - fs_ndcg):.4f}; verdicts {'AGREE' if agree else 'DISAGREE'}"
    )

    sess.sagemaker_client.delete_model(ModelName=MODEL_NAME)
    print("deleted the Clarify model; done.")


if __name__ == "__main__":
    main()
