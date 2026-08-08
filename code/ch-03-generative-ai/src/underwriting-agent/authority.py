"""The delegated-authority router: which role may approve a deal."""

import json
import pathlib

MATRIX_PATH = pathlib.Path("data/authority_limits.json")


def load_matrix(path: pathlib.Path = MATRIX_PATH) -> list[dict]:
    """Load the authority-limits matrix from disk."""
    return json.loads(path.read_text())


def decide(exposure_usd: float, risk_profile: int, tiers: list[dict]) -> dict:
    """Return the role that may approve the deal: the first tier whose ceilings both cover it."""
    for tier in tiers:
        max_exposure = tier["max_exposure_usd"]
        within_exposure = max_exposure is None or exposure_usd <= max_exposure
        within_risk = risk_profile <= tier["max_risk_profile"]
        if within_exposure and within_risk:
            return {
                "role": tier["name"],
                "tier": tier["tier"],
                "escalates": tier["tier"] > 1,
                "exposure_usd": exposure_usd,
                "risk_profile": risk_profile,
            }
    top = tiers[-1]
    return {
        "role": top["name"],
        "tier": top["tier"],
        "escalates": True,
        "exposure_usd": exposure_usd,
        "risk_profile": risk_profile,
    }
