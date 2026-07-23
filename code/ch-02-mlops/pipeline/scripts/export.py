"""Pipeline step 4 (Processing): write the winner's held-out scores to S3.

Reads winner.json (from select) and writes scores.csv to the processing output, which
SageMaker uploads to the ProcessingOutput S3 destination. Same job local or on AWS.
"""

import csv
import json
import os

IN = "/opt/ml/processing/input/winner.json"
OUT = "/opt/ml/processing/output"


def main() -> None:
    """Format the winner's scores as scores.csv."""
    with open(IN) as fh:
        winner = json.load(fh)
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/scores.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["row", "model", "pd"])
        for i, pd in enumerate(winner["scores"]):
            writer.writerow([i, winner["model"], round(float(pd), 6)])
    print(f"export: wrote {len(winner['scores'])} scores for '{winner['model']}'")


if __name__ == "__main__":
    main()
