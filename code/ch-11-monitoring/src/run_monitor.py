# /// script
# dependencies = ["catboost", "pandas", "numpy", "boto3"]
# ///
"""A monitoring job over the drift monitors, emitting the SageMaker-shaped artifacts.

This is the harness over monitor.py that a SageMaker Model Monitor + Clarify schedule
runs on AWS. It writes the Clarify explainability report (analysis.json, with global
mean-absolute SHAP values per feature and the expected value), the Model Monitor
constraint_violations.json, then publishes each metric to CloudWatch and reads back
which alarms are breached. The alarm thresholds are the drift thresholds from monitor.py.

Usage:
  MONITOR_LOCAL=1 uv run src/run_monitor.py
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from cloudwatch import get_cloudwatch
from model import FEATURES, load
from monitor import NDCG_MIN, PSI_MAJOR, _attributions, run

NAMESPACE = "ch11/monitoring"


def analysis_json(model, df: pd.DataFrame) -> dict:
    """Render Clarify's explainability report: global mean-abs SHAP per feature."""
    attr = _attributions(model, df)
    return {
        "explanations": {
            "kernel_shap": {
                "default": {
                    "global_shap_values": {
                        f: round(float(attr[f]), 4) for f in FEATURES
                    },
                    "expected_value": round(
                        float(model.predict_proba(df[FEATURES])[:, 1].mean()), 4
                    ),
                }
            }
        }
    }


def constraint_violations(report: dict) -> dict:
    """Render Model Monitor's constraint_violations.json from the monitor report."""
    return {
        "violations": [
            {
                "feature_name": v["feature"],
                "constraint_check_type": v["monitor"],
                "description": f"{v['metric']} {v['value']} breached threshold {v['threshold']}",
            }
            for v in report["violations"]
        ]
    }


def main() -> None:
    """Run the monitors, write the artifacts, publish metrics, and evaluate alarms."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--reference", type=Path, default=Path("data/generated/reference.csv")
    )
    p.add_argument("--current", type=Path, default=Path("data/generated/current.csv"))
    p.add_argument("--model", type=Path, default=Path("data/generated/scorecard.cbm"))
    p.add_argument("--out", type=Path, default=Path("outputs/monitoring"))
    a = p.parse_args()

    model = load(a.model)
    reference, current = pd.read_csv(a.reference), pd.read_csv(a.current)
    report = run(model, reference, current)

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
    cw.put_metric_data(
        Namespace=NAMESPACE,
        MetricData=[
            {"MetricName": "score_psi", "Value": report["metrics"]["score_psi"]}
        ],
    )
    cw.put_metric_data(
        Namespace=NAMESPACE,
        MetricData=[
            {
                "MetricName": "attribution_ndcg",
                "Value": report["metrics"]["attribution_ndcg"],
            }
        ],
    )
    for feature, psi in report["metrics"]["feature_psi"].items():
        cw.put_metric_data(
            Namespace=NAMESPACE,
            MetricData=[{"MetricName": f"psi_{feature}", "Value": psi}],
        )

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
        MetricName="attribution_ndcg",
        Threshold=NDCG_MIN,
        ComparisonOperator="LessThanThreshold",
    )
    for feature in report["metrics"]["feature_psi"]:
        cw.put_metric_alarm(
            AlarmName=f"ch11-psi-{feature}",
            Namespace=NAMESPACE,
            MetricName=f"psi_{feature}",
            Threshold=PSI_MAJOR,
            ComparisonOperator="GreaterThanThreshold",
        )

    status = "CompletedWithViolations" if report["violations"] else "Completed"
    print(
        f"monitoring job: {status}  ({len(report['violations'])} violations) -> {a.out}"
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
