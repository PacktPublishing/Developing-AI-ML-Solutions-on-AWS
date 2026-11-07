"""Update the challenger split percentage in the local feature-flag JSON.

The local rollout workflow converts the updated JSON to Ion and restarts the
AppConfig agent to load it.

Usage:
  uv run etl/flip.py 50
"""

import argparse
import json
import re
from pathlib import Path

FLAGS = Path("local/flags/feature-flags.json")


def main() -> None:
    """Rewrite the challenger variant's pct:: in the split rule."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("pct", type=int)
    p.add_argument("--flags", type=Path, default=FLAGS)
    a = p.parse_args()

    doc = json.loads(a.flags.read_text())
    variant = doc["challenger_rollout"]["_variants"][0]
    variant["rule"] = re.sub(r"pct::\d+", f"pct::{a.pct}", variant["rule"])
    a.flags.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"challenger rollout -> {a.pct}%  ({variant['rule']})")


if __name__ == "__main__":
    main()
