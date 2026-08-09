# /// script
# dependencies = ["boto3"]
# ///
"""The routing seam: Amazon SNS on AWS, a from-source shim locally.

get_sns() returns an SNS client exposing create_topic / subscribe / publish. On
AWS it is boto3's sns client; locally it is LocalSNS, modelled on the community
LocalStack SNS provider: topics hold subscriptions, and publish fans the message
out to every subscriber. The local delivery target is a file inbox per team, so a
classified conversation lands in that team's queue exactly as a real subscriber
(an SQS queue or Lambda) would receive it.
"""

import json
import os
from pathlib import Path

import boto3


class LocalSNS:
    """A minimal SNS stand-in: topics, subscriptions, and fan-out on publish."""

    def __init__(self, inbox: Path) -> None:
        """Start empty topic and subscription tables; deliver to files under `inbox`."""
        self._topics: set[str] = set()
        self._subs: dict[str, list[dict]] = {}
        self._inbox = inbox
        self._inbox.mkdir(parents=True, exist_ok=True)

    def create_topic(self, Name: str, **_) -> dict:
        """Create a topic (idempotent) and return its ARN."""
        arn = f"arn:aws:sns:local:000000000000:{Name}"
        self._topics.add(arn)
        self._subs.setdefault(arn, [])
        return {"TopicArn": arn}

    def subscribe(self, TopicArn: str, Protocol: str, Endpoint: str, **_) -> dict:
        """Subscribe an endpoint to a topic."""
        self._subs.setdefault(TopicArn, []).append(
            {"protocol": Protocol, "endpoint": Endpoint}
        )
        return {"SubscriptionArn": f"{TopicArn}:{len(self._subs[TopicArn])}"}

    def publish(
        self, TopicArn: str, Message: str, Subject: str | None = None, **_
    ) -> dict:
        """Fan the message out to every subscriber (here, appended to a file inbox)."""
        for sub in self._subs.get(TopicArn, []):
            path = self._inbox / f"{sub['endpoint']}.jsonl"
            with path.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "topic": TopicArn.split(":")[-1],
                            "subject": Subject,
                            "message": Message,
                        }
                    )
                    + "\n"
                )
        return {"MessageId": "local"}


def get_sns():
    """Return an SNS client: real boto3, or the local file-inbox shim."""
    if os.environ.get("BEDROCK_LOCAL") == "1":
        return LocalSNS(Path(os.environ.get("SNS_INBOX", "outputs/inbox")))
    return boto3.client(
        "sns", region_name=os.environ.get("BEDROCK_REGION", "us-east-1")
    )
