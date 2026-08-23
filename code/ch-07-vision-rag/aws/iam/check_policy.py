# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Validate deploy.json before it reaches IAM.

uv run aws/iam/check_policy.py aws/iam/deploy.json

IAM rejects a malformed document with 'Vendor is not valid', which says nothing about
which action is wrong. This names it. It also catches the specific way this file got
corrupted once: an Action that is a bare string, mutated as if it were a list, leaving
one statement holding the individual characters of 'logs:'.
"""

import json
import re
import sys
from pathlib import Path

ACTION = re.compile(r"^[a-z0-9-]+:[A-Za-z0-9*]+$")


def main() -> None:
    """Report every malformed action, and exit non-zero if there are any."""
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "aws/iam/deploy.json")
    doc = json.loads(path.read_text())

    problems = []
    for statement in doc["Statement"]:
        sid = statement.get("Sid", "<no Sid>")
        actions = statement["Action"]
        actions = actions if isinstance(actions, list) else [actions]
        for action in actions:
            if action != "*" and not ACTION.match(action):
                problems.append(f"  {sid}: {action!r}")
        if len(actions) > 4 and all(len(a) <= 2 for a in actions):
            problems.append(f"  {sid}: looks like a string was iterated as characters")

    if problems:
        print(f"{path}: malformed actions", *problems, sep="\n")
        raise SystemExit(1)
    print(f"{path}: {len(doc['Statement'])} statements, all actions well formed")


if __name__ == "__main__":
    main()
