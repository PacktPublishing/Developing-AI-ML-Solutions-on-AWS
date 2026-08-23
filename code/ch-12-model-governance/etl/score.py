"""Route a batch of applications through the gateway and summarise the split.

POSTs each application to the gateway's /score and tallies which variant served it and
what it decided -- the view an operator watches while a rollout widens.

Usage:
  uv run etl/score.py data/applications.jsonl
"""

import argparse
import json
import os
import urllib.request
from collections import Counter
from pathlib import Path

GATEWAY = os.environ.get("GATEWAY_URL", "http://localhost:8080/score")


def main() -> None:
    """Score every application and print the variant and decision tallies."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("applications", type=Path)
    a = p.parse_args()

    variants: Counter = Counter()
    decisions: Counter = Counter()
    for line in a.applications.read_text().splitlines():
        if not line.strip():
            continue
        req = urllib.request.Request(
            GATEWAY, data=line.encode(), headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read())
        variants[body["variant"]] += 1
        decisions[body["decision"]] += 1

    total = sum(variants.values())
    print(f"routed {total} applications through the gateway")
    print(
        "  variants:  "
        + ", ".join(f"{v} {n} ({n / total:.0%})" for v, n in variants.items())
    )
    print("  decisions: " + ", ".join(f"{d} {n}" for d, n in decisions.items()))


if __name__ == "__main__":
    main()
