"""Pipeline step 3 (Processing): compare the two models and pick the winner.

Reads each model's metrics.json (from the two train steps) and writes the higher-AUC
one to winner.json (ties go to the transparent scorecard).
"""

import json

SCORECARD = "/opt/ml/processing/scorecard/metrics.json"
CHALLENGER = "/opt/ml/processing/challenger/metrics.json"
OUT = "/opt/ml/processing/winner/winner.json"


def main() -> None:
    """Select the higher-AUC model and stage it for the export step."""
    import os

    with open(SCORECARD) as fh:
        scorecard = json.load(fh)
    with open(CHALLENGER) as fh:
        challenger = json.load(fh)
    winner = challenger if challenger["auc"] > scorecard["auc"] else scorecard

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(winner, fh)
    print(
        f"select: winner={winner['model']} (scorecard {scorecard['auc']} vs challenger {challenger['auc']})"
    )


if __name__ == "__main__":
    main()
