"""The drift monitors as a SageMaker Processing job -- AWS's replacement for Clarify.

SageMaker Clarify is closed to new accounts; AWS's own guidance is to run the same
standardized metrics and SHAP feature attribution yourself as a Processing job. That
is this file: it satisfies the Processing path contract, so the one monitor image runs
in local mode and as a managed job alike.

  /opt/ml/processing/input/reference/reference.csv   the batch the model trained on
  /opt/ml/processing/input/current/current.csv       the live batch to compare
  /opt/ml/processing/input/model/scorecard.cbm       the model under watch
  /opt/ml/processing/output/                          analysis.json + violations

It computes the PSI and SHAP-attribution drift with monitor.py, writes the
Clarify-shaped artifacts, and publishes the drift metrics to CloudWatch, where the
stack's alarms turn them into alerts.
"""

import json
import os

import boto3
import pandas as pd
from model import load
from monitor import run
from run_monitor import NAMESPACE, analysis_json, constraint_violations

PREFIX = "/opt/ml/processing"
REFERENCE = f"{PREFIX}/input/reference/reference.csv"
CURRENT = f"{PREFIX}/input/current/current.csv"
MODEL = f"{PREFIX}/input/model/scorecard.cbm"
SCORES = f"{PREFIX}/input/scores"
OUTPUT = f"{PREFIX}/output"


def _transform_scores(current: pd.DataFrame) -> pd.Series | None:
    """Read the Batch Transform predictions for the current batch, if the step ran."""
    if not os.path.isdir(SCORES):
        return None
    out = [f for f in sorted(os.listdir(SCORES)) if f.endswith(".out")]
    if not out:
        return None
    scores = pd.concat(
        [pd.read_csv(os.path.join(SCORES, f), header=None)[0] for f in out],
        ignore_index=True,
    )
    return pd.Series(scores.to_numpy(), index=current.index, name="score")


def main() -> None:
    """Run the monitors, write the artifacts, and publish the drift metrics."""
    model = load(MODEL)
    reference, current = pd.read_csv(REFERENCE), pd.read_csv(CURRENT)
    report = run(model, reference, current, current_scores=_transform_scores(current))

    os.makedirs(OUTPUT, exist_ok=True)
    for name, payload in [
        ("baseline_analysis.json", analysis_json(model, reference)),
        ("analysis.json", analysis_json(model, current)),
        ("constraint_violations.json", constraint_violations(report)),
    ]:
        with open(os.path.join(OUTPUT, name), "w") as fh:
            json.dump(payload, fh, indent=2)

    metrics = report["metrics"]
    data = [
        {
            "MetricName": "feature_attribution_ndcg",
            "Value": metrics["attribution_ndcg"],
        },
        {"MetricName": "score_psi", "Value": metrics["score_psi"]},
        *(
            {"MetricName": f"psi_{feature}", "Value": psi}
            for feature, psi in metrics["feature_psi"].items()
        ),
    ]
    region = os.environ.get(
        "AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    )
    boto3.client("cloudwatch", region_name=region).put_metric_data(
        Namespace=NAMESPACE, MetricData=data
    )

    status = "CompletedWithViolations" if report["violations"] else "Completed"
    print(f"monitoring job: {status}  ({len(report['violations'])} violations)")
    print(
        f"  score_psi={metrics['score_psi']}  attribution_ndcg={metrics['attribution_ndcg']}"
    )


if __name__ == "__main__":
    main()
