# /// script
# dependencies = []
# ///
"""The auditor's record: model cards, a registry, and lineage from model metadata.

Reads governance/models.json -- the facts a team owns about each model -- and renders
the three artifacts an audit reads: a SageMaker-shaped model card (overview, intended
uses and risk rating, training and evaluation details), a model registry entry (the
model package group, its version, and the champion/challenger stage), and a lineage
record (dataset -> training -> model -> deployment). The same JSON is what the AWS round
uploads to SageMaker Model Cards, the Model Registry, and ML Lineage.

Usage:
  uv run src/governance.py --models governance/models.json --out outputs/governance
"""

import argparse
import json
from pathlib import Path


def model_card(name: str, meta: dict) -> dict:
    """Render a SageMaker-shaped model card from a model's metadata."""
    metrics = meta["metrics"]
    return {
        "model_overview": {
            "model_name": name,
            "model_description": meta["description"],
            "algorithm_type": meta["algorithm"],
            "problem_type": meta["problem_type"],
            "model_owner": meta["owner"],
        },
        "intended_uses": {
            "purpose_of_model": meta["intended_use"],
            "intended_uses": meta["intended_use"],
            "factors_affecting_model_efficiency": "Trained on the chapter's synthetic "
            "credit-bureau population; efficiency degrades on populations that drift "
            "from it (see the monitoring chapter).",
            "risk_rating": meta["risk_rating"],
            "explanations_for_risk_rating": meta["risk_rationale"],
            "out_of_scope_uses": meta["out_of_scope"],
        },
        "training_details": {
            "objective_function": "Probability of default (binary log loss)",
            "training_datasets": [meta["dataset"]],
            "training_features": meta["features"],
        },
        "evaluation_details": [
            {
                "name": meta["metrics_source"],
                "metric_groups": [
                    {
                        "name": "discrimination",
                        "metric_data": [
                            {"name": k, "type": "number", "value": v}
                            for k, v in metrics.items()
                        ],
                    }
                ],
            }
        ],
        "additional_information": {
            "ethical_considerations": "Credit decisions are high-impact; the model is "
            "one input to a human-reviewed decision, not an automated adverse action.",
        },
    }


def registry(models: dict) -> dict:
    """Render a registry: one model package group per model, with its stage."""
    return {
        "model_package_groups": [
            {
                "model_package_group_name": name,
                "stage": meta["stage"],
                "latest_version": 1,
                "model_card": f"{name}-card.json",
                "lineage": f"{name}-lineage.json",
                "metrics": meta["metrics"],
            }
            for name, meta in models.items()
        ]
    }


def lineage(name: str, meta: dict) -> dict:
    """Render a lineage record: dataset -> training -> model -> deployment."""
    return {
        "model": name,
        "associations": [
            {"source": meta["dataset"], "type": "DataSet", "action": "ContributedTo"},
            {"source": f"train-{name}", "type": "TrainingJob", "action": "Produced"},
            {"source": f"{name}:1", "type": "Model", "action": "DeployedTo"},
            {
                "source": "credit-rollout-gateway",
                "type": "Endpoint",
                "action": "Serves",
            },
        ],
    }


def main() -> None:
    """Write a card, a registry entry, and a lineage record for every model."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", type=Path, default=Path("governance/models.json"))
    p.add_argument("--out", type=Path, default=Path("outputs/governance"))
    a = p.parse_args()

    models = json.loads(a.models.read_text())
    a.out.mkdir(parents=True, exist_ok=True)
    for name, meta in models.items():
        (a.out / f"{name}-card.json").write_text(
            json.dumps(model_card(name, meta), indent=2)
        )
        (a.out / f"{name}-lineage.json").write_text(
            json.dumps(lineage(name, meta), indent=2)
        )
    (a.out / "registry.json").write_text(json.dumps(registry(models), indent=2))
    print(
        f"wrote model cards, registry, and lineage for {len(models)} models -> {a.out}"
    )


if __name__ == "__main__":
    main()
