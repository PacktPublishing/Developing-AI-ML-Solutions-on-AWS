# /// script
# dependencies = ["catboost", "pandas", "numpy", "boto3"]
# ///
"""The monitoring job over the drift monitors, emitting the SageMaker-shaped artifacts.

One entrypoint for both environments. Locally it runs on the generated batches,
publishes to the CloudWatch shim, and bootstraps the alarms, because no stack exists
to own them. As the SageMaker Processing step the pipeline passes the
/opt/ml/processing paths as arguments (--scores carries the Batch Transform
predictions) and the metrics go to real CloudWatch, where the stack's alarms watch
the same names.

Usage:
  MONITOR_LOCAL=1 uv run src/run_monitor.py
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from attribution import NDCG_MIN, analysis_json
from cloudwatch import NAMESPACE, get_cloudwatch
from model import load
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
    p.add_argument("--out", type=Path, default=Path("outputs/monitoring"))
    a = p.parse_args()

    model = load(a.model)
    reference, current = pd.read_csv(a.reference), pd.read_csv(a.current)
    report = run(
        model, reference, current, current_scores=transform_scores(a.scores, current)
    )

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
