# /// script
# dependencies = ["boto3"]
# ///
"""The alerting seam: Amazon CloudWatch on AWS, a from-source shim locally.

get_cloudwatch() returns a client exposing put_metric_data, put_metric_alarm, and
describe_alarms. On AWS it is boto3's cloudwatch client; locally it is a shim,
modelled on the community LocalStack CloudWatch provider: it stores metric data and
evaluates each alarm's latest datapoint against its threshold, so a monitoring job
can publish PSI and attribution metrics and raise an alarm before bad scores pile up.
"""

import os

import boto3

_OPS = {
    "GreaterThanThreshold": lambda v, t: v > t,
    "GreaterThanOrEqualToThreshold": lambda v, t: v >= t,
    "LessThanThreshold": lambda v, t: v < t,
    "LessThanOrEqualToThreshold": lambda v, t: v <= t,
}


class LocalCloudWatch:
    """A CloudWatch stand-in: a metric store plus threshold-evaluated alarms."""

    def __init__(self) -> None:
        """Start empty metric and alarm tables."""
        self._metrics: dict[tuple[str, str], float] = {}
        self._alarms: dict[str, dict] = {}

    def put_metric_data(self, Namespace: str, MetricData: list, **_) -> dict:
        """Record the latest value of each metric datapoint."""
        for d in MetricData:
            self._metrics[(Namespace, d["MetricName"])] = float(d["Value"])
        return {}

    def put_metric_alarm(
        self,
        AlarmName: str,
        Namespace: str,
        MetricName: str,
        Threshold: float,
        ComparisonOperator: str,
        **_,
    ) -> dict:
        """Define an alarm over a metric and a threshold."""
        self._alarms[AlarmName] = {
            "Namespace": Namespace,
            "MetricName": MetricName,
            "Threshold": float(Threshold),
            "ComparisonOperator": ComparisonOperator,
        }
        return {}

    def describe_alarms(self, **_) -> dict:
        """Evaluate every alarm against its metric's latest value; return their states."""
        out = []
        for name, a in self._alarms.items():
            value = self._metrics.get((a["Namespace"], a["MetricName"]))
            breached = value is not None and _OPS[a["ComparisonOperator"]](
                value, a["Threshold"]
            )
            out.append(
                {
                    "AlarmName": name,
                    "StateValue": "ALARM" if breached else "OK",
                    "MetricName": a["MetricName"],
                    "value": value,
                    "Threshold": a["Threshold"],
                }
            )
        return {"MetricAlarms": out}


def get_cloudwatch():
    """Return a CloudWatch client: real boto3, or the local shim."""
    if os.environ.get("MONITOR_LOCAL") == "1":
        return LocalCloudWatch()
    return boto3.client(
        "cloudwatch", region_name=os.environ.get("AWS_REGION", "us-east-1")
    )
