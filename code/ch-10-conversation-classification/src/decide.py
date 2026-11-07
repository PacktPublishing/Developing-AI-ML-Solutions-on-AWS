# /// script
# dependencies = ["boto3", "strands-decider"]
# ///
"""Classify conversations using a decision model and write prediction records.

Each response contains a selected team, class probabilities and confidence.
Clear the prediction when confidence is below the configured threshold. The
output uses the format accepted by evaluate.py.

Usage:
  DECIDER_LOCAL=1 PYTHONPATH=src uv run src/decide.py --variant b --limit 40
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import categories
from decider import get_decider
from runs import new_run_id

ASK = "Which team should handle this conversation?"


def question_for(labels: list[str], described: bool) -> dict:
    """Build a choice question using the supplied category labels."""
    return {
        "team": {
            "type": "choice",
            "instructions": ASK,
            "criteria": categories.as_criteria(labels, described),
        }
    }


def route_one(decider, text: str, question: dict) -> tuple[str, dict, float, str]:
    """Return the team, the distribution over teams, the confidence, and the model.

    The model comes back from the backend rather than from configuration, so a
    predictions file records which model actually answered. That is what lets a
    Strands Decider run and a Kev run be compared after the fact.
    """
    response = decider.decide(text, question)
    answer = response["answers"]["team"]
    return (
        answer["choice"],
        answer["probabilities"],
        answer["confidence"],
        str(response.get("model") or "unnamed"),
    )


def record_for(
    row: dict,
    team: str,
    probs: dict,
    confidence: float,
    threshold: float,
    meta: dict,
) -> dict:
    """Build a prediction record, clearing predictions below the confidence threshold."""
    record = {
        "id": row["id"],
        "label": row["label"],
        "pred": team,
        "probs": probs,
        "confidence": confidence,
        **meta,
    }
    if confidence < threshold:
        record["pred"] = ""
        record["probs"] = None
        record["failed"] = (
            f"confidence {confidence:.3f} below the {threshold:.2f} routing threshold"
        )
    return record


def main() -> None:
    """Route a prepared dataset with the decision model and write predictions."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="conversations")
    p.add_argument("--data", type=Path, default=Path("data/generated"))
    p.add_argument("--limit", type=int, default=0)
    p.add_argument(
        "--variant",
        default="b",
        choices=["a", "b"],
        help="a: bare team names; b: the team definitions from categories.py",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="confidence below which a conversation is left for a human (0 routes all)",
    )
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()

    rows = [
        json.loads(line)
        for line in (a.data / f"{a.dataset}.jsonl").read_text().splitlines()
    ]
    if a.limit:
        rows = rows[: a.limit]
    labels = json.loads((a.data / f"{a.dataset}.labels.json").read_text())

    question = question_for(labels, described=a.variant == "b")
    run_id = new_run_id()
    decider = get_decider()
    print(f"decision model on {decider.where}, variant {a.variant}, {len(rows)} rows")

    out = a.out or a.data / f"{a.dataset}.predictions.jsonl"
    latencies = []
    hits = held = 0
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            started = time.perf_counter()
            team, probs, confidence, model = route_one(decider, row["text"], question)
            latencies.append(time.perf_counter() - started)
            meta = {"run_id": run_id, "model": model, "backend": decider.where}
            record = record_for(row, team, probs, confidence, a.threshold, meta)
            if not record["pred"]:
                held += 1
            elif team == row["label"]:
                hits += 1
            f.write(json.dumps(record) + "\n")

    routed = len(rows) - held
    print(
        f"{hits}/{routed} of the routed conversations correct, "
        f"{held} left for a human -> {out}"
    )
    print(
        f"  median {statistics.median(latencies) * 1000:.0f} ms per conversation, "
        f"{sum(latencies):.1f}s for {len(rows)}"
    )


if __name__ == "__main__":
    main()
