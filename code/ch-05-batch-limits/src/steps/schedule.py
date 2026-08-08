# /// script
# requires-python = ">=3.12"
# dependencies = ["boto3"]
# ///
"""Create the schedule and the alerts topic, same shapes as aws/template.yaml.

Usage (with `uv run events-local/app.py` serving in another terminal):
  uv run steps/schedule.py
"""

import os

import boto3

REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
EVENTS_ENDPOINT = os.environ.get("EVENTS_ENDPOINT", "http://localhost:8009")
PIPELINE_ARN = f"arn:aws:sagemaker:{REGION}:000000000000:pipeline/batch-credit-limits"
SCHEDULE = os.environ.get("SCHEDULE_EXPRESSION", "rate(1 minute)")


def main() -> None:
    """Create the alerts topic, subscribe, and schedule the pipeline rule."""
    sns = boto3.client("sns", region_name=REGION, endpoint_url=EVENTS_ENDPOINT)
    topic = sns.create_topic(Name="ch05-batch-alerts")["TopicArn"]
    sns.subscribe(TopicArn=topic, Protocol="email", Endpoint="analyst@example.com")

    events = boto3.client("events", region_name=REGION, endpoint_url=EVENTS_ENDPOINT)
    events.put_rule(Name="batch-schedule", ScheduleExpression=SCHEDULE, State="ENABLED")
    events.put_targets(
        Rule="batch-schedule",
        Targets=[{"Id": "batch-pipeline", "Arn": PIPELINE_ARN}],
    )
    print(f"topic {topic}; rule batch-schedule ({SCHEDULE}) -> {PIPELINE_ARN}")


if __name__ == "__main__":
    main()
