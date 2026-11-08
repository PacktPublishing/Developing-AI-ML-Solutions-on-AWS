# /// script
# dependencies = ["catboost", "pandas", "numpy", "boto3", "botocore[crt]"]
# ///
"""Run drift monitoring, write JSON reports and publish CloudWatch metrics.

Read predictions from Batch Transform output or endpoint capture records.
MONITOR_LOCAL selects the local CloudWatch client and creates local alarms.
On AWS, the stack defines the alarms and this script publishes their metrics.

Usage:
  MONITOR_LOCAL=1 uv run src/run_monitor.py
"""

import argparse
import json
import os
from pathlib import Path

import bias
import pandas as pd
from attribution import NDCG_MIN, analysis_json
from cloudwatch import NAMESPACE, get_cloudwatch
from model import load, score
from monitor import constraint_violations, run
from psi import PSI_MAJOR


def transform_scores(
    scores_dir: Path | None, current: pd.DataFrame
) -> pd.Series | None:
    """Read the Batch Transform predictions for the current batch, if the step ran."""
    if scores_dir is None or not scores_dir.is_dir():
        return None
    out = sorted(f for f in scores_dir.iterdir() if f.name.endswith(".out"))
    if not out:
        return None
    scores = pd.concat([pd.read_csv(f, header=None)[0] for f in out], ignore_index=True)
    return pd.Series(scores.to_numpy(), index=current.index, name="score")


def bias_for(
    current: pd.DataFrame,
    scores: pd.Series | None,
    facets: list[str],
    label_column: str,
    threshold: float,
) -> list[bias.Report]:
    """Compute the bias metrics for the current batch, per facet.

    Captured endpoint traffic has no outcomes in it, so the label column is only
    used when the current frame actually carries one. Without it the report is
    the three metrics that read predictions alone.
    """
    if scores is None:
        return []
    frame = current.copy()
    frame["_predicted"] = (scores.to_numpy() >= threshold).astype(int)
    label = label_column if label_column in frame.columns else None
    return bias.reports_for(frame, facets, label, "_predicted")


def log_bias_to_mlflow(reports: list[bias.Report]) -> None:
    """Log each facet's metrics to an MLflow App, when one is configured.

    AWS's guidance for replacing Clarify asks for the per-segment results to
    carry the same lineage and versioning as the training runs, which is what
    the MLflow App is for. No tracking URI means nothing to log to, and the
    metrics have already gone to CloudWatch either way.
    """
    if not os.environ.get("MLFLOW_TRACKING_URI") or not reports:
        return
    import mlflow  # deferred: only this path needs it

    with mlflow.start_run(run_name="bias-monitoring"):
        for r in reports:
            facet = f"{r.facet}={r.value}"
            mlflow.log_param(f"facet.{r.facet}", r.value)
            mlflow.log_metric(f"n_disfavored.{r.facet}", r.n_disfavored)
            for name, value in r.metrics.items():
                # MLflow rejects a non-finite metric, and DI and TE are both
                # defined as infinite when their denominator is zero.
                if value in (float("inf"), float("-inf")):
                    continue
                mlflow.log_metric(f"{name}.{facet}", value)


def main() -> None:
    """Run the monitors, write the artifacts, publish metrics, and evaluate alarms."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--reference", type=Path, default=Path("data/generated/reference.csv")
    )
    p.add_argument("--current", type=Path, default=Path("data/generated/current.csv"))
    p.add_argument("--model", type=Path, default=Path("runs-local/model/scorecard.cbm"))
    p.add_argument(
        "--scores",
        type=Path,
        default=None,
        help="directory of Batch Transform .out predictions for the current batch",
    )
    p.add_argument(
        "--capture",
        default=None,
        help="SageMaker Data Capture records (local dir/file or s3:// prefix) to monitor "
        "live endpoint traffic instead of --current + --scores",
    )
    p.add_argument("--out", type=Path, default=Path("outputs/monitoring"))
    # Clarify is closed to new customers, and AWS's replacement guidance says to
    # compute these metrics directly and log them per segment. A facet is written
    # column=value and names the disfavored group.
    p.add_argument(
        "--facet",
        action="append",
        default=None,
        help="a facet as column=value; repeat for more than one",
    )
    p.add_argument("--label", default="default", help="the outcome column, where known")
    p.add_argument(
        "--decision-threshold",
        type=float,
        default=0.5,
        help="the score above which a prediction counts as positive",
    )
    a = p.parse_args()

    model = load(a.model)
    reference = pd.read_csv(a.reference)
    if a.capture:
        # live path: reconstruct the current batch and its scores from captured traffic
        from capture import read_capture

        current, current_scores = read_capture(a.capture)
    else:
        # batch path: the current CSV, scored by the Batch Transform step if it ran
        current = pd.read_csv(a.current)
        current_scores = transform_scores(a.scores, current)
    report = run(model, reference, current, current_scores=current_scores)

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "baseline_analysis.json").write_text(
        json.dumps(analysis_json(model, reference), indent=2)
    )
    (a.out / "analysis.json").write_text(
        json.dumps(analysis_json(model, current), indent=2)
    )
    (a.out / "constraint_violations.json").write_text(
        json.dumps(constraint_violations(report), indent=2)
    )

    facets = a.facet or ["employment_status=unemployed", "home_ownership=RENT"]
    # The bias metrics need a prediction per row. Batch Transform's output is the
    # one to use when the step ran, because it is what the endpoint would have
    # said; without it the model scores the frame here so the local path reports
    # the same metrics.
    for_bias = current_scores if current_scores is not None else score(model, current)
    bias_reports = bias_for(current, for_bias, facets, a.label, a.decision_threshold)
    (a.out / "bias.json").write_text(
        json.dumps([vars(r) for r in bias_reports], indent=2)
    )

    cw = get_cloudwatch()
    metrics = report["metrics"]
    for name, value in [
        ("score_psi", metrics["score_psi"]),
        ("feature_attribution_ndcg", metrics["attribution_ndcg"]),
        *((f"psi_{f}", v) for f, v in metrics["feature_psi"].items()),
    ]:
        cw.put_metric_data(
            Namespace=NAMESPACE, MetricData=[{"MetricName": name, "Value": value}]
        )

    status = "CompletedWithViolations" if report["violations"] else "Completed"
    print(
        f"monitoring job: {status}  ({len(report['violations'])} violations) -> {a.out}"
    )
    print(
        f"  score_psi={metrics['score_psi']}  attribution_ndcg={metrics['attribution_ndcg']}"
    )
    for r in bias_reports:
        n = r.n_disfavored + r.n_favored
        shown = (
            " ".join(f"{k}={v:+.4f}" for k, v in sorted(r.metrics.items()))
            or "no rows matched this facet, so it has no metrics"
        )
        print(f"  {r.facet}={r.value} (n={r.n_disfavored}/{n}) {shown}")
    for r in bias_reports:
        for name, value in r.metrics.items():
            cw.put_metric_data(
                Namespace=NAMESPACE,
                MetricData=[
                    {
                        "MetricName": f"bias_{name}",
                        "Value": value,
                        "Dimensions": [
                            {"Name": "Facet", "Value": f"{r.facet}={r.value}"}
                        ],
                    }
                ],
            )
    log_bias_to_mlflow(bias_reports)

    if os.environ.get("MONITOR_LOCAL") != "1":
        return
    # no stack owns the alarms locally, so the job bootstraps and evaluates them;
    # on AWS template.yaml owns them and the job only publishes the metrics
    cw.put_metric_alarm(
        AlarmName="ch11-score-drift",
        Namespace=NAMESPACE,
        MetricName="score_psi",
        Threshold=PSI_MAJOR,
        ComparisonOperator="GreaterThanThreshold",
    )
    cw.put_metric_alarm(
        AlarmName="ch11-attribution-drift",
        Namespace=NAMESPACE,
        MetricName="feature_attribution_ndcg",
        Threshold=NDCG_MIN,
        ComparisonOperator="LessThanThreshold",
    )
    for feature in metrics["feature_psi"]:
        cw.put_metric_alarm(
            AlarmName=f"ch11-psi-{feature}",
            Namespace=NAMESPACE,
            MetricName=f"psi_{feature}",
            Threshold=PSI_MAJOR,
            ComparisonOperator="GreaterThanThreshold",
        )
    alarms = [
        al for al in cw.describe_alarms()["MetricAlarms"] if al["StateValue"] == "ALARM"
    ]
    print(f"CloudWatch alarms firing ({len(alarms)}):")
    for al in alarms:
        print(
            f"  ALARM {al['AlarmName']:26} {al['MetricName']} = {al['value']} (threshold {al['Threshold']})"
        )


if __name__ == "__main__":
    main()
