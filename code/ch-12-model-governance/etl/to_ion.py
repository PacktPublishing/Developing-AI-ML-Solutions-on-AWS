"""Render the feature-flag JSON as the Amazon Ion the AppConfig agent evaluates.

The agent only evaluates rules (including the split) when the configuration is Ion of
type AWS.AppConfig.FeatureFlags, in a file named app:env:profile.application%ion%type=
AWS.AppConfig.FeatureFlags. This converts our JSON flag document to that Ion form, one
source of truth for both the local agent and the AWS hosted configuration version.

Usage:
  uv run etl/to_ion.py --app credit-governance --env local --profile rollout
"""

import argparse
import json
from pathlib import Path


def to_ion(flags: dict) -> str:
    """Render the flag document as Ion: each flag a list of [rule, variant] and a default."""
    blocks = []
    for flag_name, flag in flags.items():
        parts = []
        for variant in flag.get("_variants", []):
            body = {
                "_variant": variant["name"],
                "enabled": variant.get("enabled", True),
            }
            body |= variant.get("attributeValues", variant.get("attributes", {}))
            body_json = json.dumps(body)
            if rule := variant.get("rule"):
                parts.append(f"   [\n     {rule},\n     '''{body_json}'''\n   ]")
            else:
                parts.append(f"   '''{body_json}'''")
        blocks.append(f"'{flag_name}'::[\n" + ",\n".join(parts) + "\n]")
    return "\n".join(blocks) + "\n"


def main() -> None:
    """Write the Ion feature-flag file the agent reads in local development mode."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--flags", type=Path, default=Path("local/flags/feature-flags.json"))
    p.add_argument("--out", type=Path, default=Path("local/agent-configs"))
    p.add_argument("--app", default="credit-governance")
    p.add_argument("--env", default="local")
    p.add_argument("--profile", default="rollout")
    a = p.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    name = (
        f"{a.app}:{a.env}:{a.profile}.application%ion%type=AWS.AppConfig.FeatureFlags"
    )
    (a.out / name).write_text(to_ion(json.loads(a.flags.read_text())))
    print(f"wrote {a.out / name}")


if __name__ == "__main__":
    main()
