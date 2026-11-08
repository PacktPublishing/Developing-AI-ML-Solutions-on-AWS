# /// script
# dependencies = ["boto3", "botocore[crt]"]
# ///
"""Provide an AWS SNS client or a local file-based implementation.

The local client stores topics and subscriptions in memory and appends
notification envelopes to subscriber inbox files.
"""

import json
import os
import uuid
from datetime import datetime, timezone
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
        """Append a notification envelope to each subscriber file and return its message ID."""
        message_id = str(uuid.uuid4())
        timestamp = (
            datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        )
        notification = {
            "Type": "Notification",
            "MessageId": message_id,
            "TopicArn": TopicArn,
            "Message": Message,
            "Timestamp": timestamp,
        }
        if Subject is not None:
            notification["Subject"] = Subject
        for sub in self._subs.get(TopicArn, []):
            path = self._inbox / f"{sub['endpoint']}.jsonl"
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(notification) + "\n")
        return {"MessageId": message_id}


def get_sns():
    """Return an SNS client: real boto3, or the local file-inbox shim."""
    if os.environ.get("BEDROCK_LOCAL") == "1":
        return LocalSNS(Path(os.environ.get("SNS_INBOX", "outputs/inbox")))
    return boto3.client(
        "sns", region_name=os.environ.get("BEDROCK_REGION", "us-east-1")
    )
