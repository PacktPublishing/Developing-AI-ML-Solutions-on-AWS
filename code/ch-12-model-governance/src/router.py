"""Route applications to the model selected by an AppConfig feature flag.

Pass loanId as context when evaluating challenger_rollout. Score the application
with the selected model and apply the decision threshold. Routing is repeatable
for a fixed identifier and configuration; configuration changes can alter it.

Usage:
  APPCONFIG_AGENT_URL=http://localhost:2772 CHAMPION_URL=... CHALLENGER_URL=...     uv run src/router.py applications.jsonl
"""

import argparse
import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from appconfig import get_appconfig
from models import http_scorer

ROLLOUT_FLAG = "challenger_rollout"
REFER_THRESHOLD = 0.5

ScoreFn = Callable[[str, dict], float]


class FlagClient(Protocol):
    """What the router needs from a feature-flag client."""

    def configuration(self, context: dict) -> dict:
        """Return the evaluated flags for a request context."""
        ...


class Router:
    """Route an application to the champion or challenger the rollout flag picks."""

    def __init__(self, appconfig: FlagClient, score: ScoreFn) -> None:
        """Take a feature-flag client and a score(model_name, loan) callable."""
        self._appconfig = appconfig
        self._score = score

    def route(self, loan: dict) -> dict:
        """Pick the model for this loan, score it, and return the decision."""
        flag = self._appconfig.configuration({"loanId": str(loan["loanId"])})[
            ROLLOUT_FLAG
        ]
        # Use the champion when the evaluated flag omits the model attribute.
        model = flag.get("model", "credit-scorecard")
        pd = self._score(model, loan)
        return {
            "loanId": loan["loanId"],
            "variant": flag["_variant"],
            "model": model,
            "pd": round(pd, 4),
            "decision": "refer" if pd >= REFER_THRESHOLD else "approve",
        }


def main() -> None:
    """Route every application in a JSONL file and summarize the split."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("applications", type=Path)
    a = p.parse_args()

    router = Router(get_appconfig(), http_scorer())
    variants: Counter = Counter()
    for line in a.applications.read_text().splitlines():
        if not line.strip():
            continue
        decision = router.route(json.loads(line))
        variants[decision["variant"]] += 1
        print(json.dumps(decision))
    total = sum(variants.values())
    print(f"\nrouted {total}: " + ", ".join(f"{v} {n}" for v, n in variants.items()))


if __name__ == "__main__":
    main()
