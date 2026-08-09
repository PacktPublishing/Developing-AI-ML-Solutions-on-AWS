# /// script
# dependencies = []
# ///
"""Widen (or narrow) the challenger rollout by editing the feature flag.

Sets the challenger variant's split percentage in the local flag store. The gateway
reads the flag on every request, so the new share takes effect on the next application
with nothing restarted -- the point of shipping behind a flag.

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
