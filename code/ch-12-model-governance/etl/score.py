"""Send applications to the gateway and count variants and decisions.

POST each JSONL record to /score and print the aggregate counts.

Usage:
  uv run etl/score.py data/applications.jsonl
"""

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

GATEWAY = os.environ.get("GATEWAY_URL", "http://localhost:8080/score")


def ask(payload: bytes, timeout: int = 30) -> dict:
    """Put one application to the gateway and read the decision back."""
    req = urllib.request.Request(
        GATEWAY, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def warm(payload: bytes, tries: int = 10) -> None:
    """Keep asking until the models answer, or give up saying why.

    A SageMaker endpoint reports InService once its container is running, which
    is not the same as the model being loaded: these load on the first request,
    and that request can outlast the serving worker's timeout. The gateway then
    returns 500 and the run looks broken when it is only cold. Nothing here can
    wait on a condition the service does not report, so this asks until an
    answer comes back.
    """
    for attempt in range(1, tries + 1):
        try:
            ask(payload, timeout=90)
            if attempt > 1:
                print(f"  gateway answered on attempt {attempt}")
            return
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as err:
            if attempt == tries:
                raise SystemExit(
                    f"the gateway did not answer after {tries} attempts: {err}. "
                    "docker logs, or the endpoint's CloudWatch log group, will "
                    "say whether the model loaded."
                ) from err
            time.sleep(15)


def main() -> None:
    """Score every application and print the variant and decision tallies."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("applications", type=Path)
    a = p.parse_args()

    variants: Counter = Counter()
    decisions: Counter = Counter()
    lines = [ln for ln in a.applications.read_text().splitlines() if ln.strip()]
    if lines:
        warm(lines[0].encode())
    for line in lines:
        body = ask(line.encode())
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
