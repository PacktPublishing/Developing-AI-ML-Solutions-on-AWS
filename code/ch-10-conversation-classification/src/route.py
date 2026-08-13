# /// script
# dependencies = ["boto3"]
# ///
"""Route each classified conversation to its team through SNS.

One topic per category; each team subscribes to its topic. Every classified
conversation is published to the topic of its predicted category, so it lands in
that team's queue. Locally the SNS shim delivers to a file inbox per team; on AWS
the same calls hit real SNS topics.

Usage:
  BEDROCK_LOCAL=1 uv run src/route.py --dataset conversations
"""

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

from sns import get_sns


def slug(name: str) -> str:
    """Turn a category name into a valid SNS topic name."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def main() -> None:
    """Publish each prediction to its category topic and print the routing summary."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="conversations")
    p.add_argument("--data", type=Path, default=Path("data/generated"))
    a = p.parse_args()

    labels = json.loads((a.data / f"{a.dataset}.labels.json").read_text())
    preds = [
        json.loads(line)
        for line in (a.data / f"{a.dataset}.predictions.jsonl").read_text().splitlines()
    ]

    sns = get_sns()
    topics = {}
    for label in labels:
        name = slug(label)
        arn = sns.create_topic(Name=f"ch10-{name}")["TopicArn"]
        # Locally the shim delivers to a file inbox per team; on AWS each team owns
        # its own subscription (an SQS queue), so we only create topics and publish.
        if os.environ.get("BEDROCK_LOCAL") == "1":
            sns.subscribe(TopicArn=arn, Protocol="file", Endpoint=f"{name}-team")
        topics[label] = arn

    routed = Counter()
    unknown = 0
    for r in preds:
        arn = topics.get(r["pred"])
        if arn is None:
            unknown += 1
            continue
        sns.publish(
            TopicArn=arn,
            Subject="new conversation",
            Message=json.dumps({"id": r["id"], "category": r["pred"]}),
        )
        routed[r["pred"]] += 1

    print("routed conversations by team:")
    for label in labels:
        print(f"  {label:22} {routed[label]}")
    if unknown:
        print(f"  (unroutable, category not recognized: {unknown})")


if __name__ == "__main__":
    main()
