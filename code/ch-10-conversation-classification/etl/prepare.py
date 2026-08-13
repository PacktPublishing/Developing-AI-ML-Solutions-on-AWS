# /// script
# dependencies = ["datasets"]
# ///
"""Build the labeled datasets the classifier files and the eval keeps honest.

RetailBanking-Conversations is the main set: rows are single turns, so we group
them by conversation, order by sequence, and render a transcript labeled with its
topic (one of ten teams). banking77 is the harder eval set: short single queries
over 77 intent classes. Both land as JSONL of {id, text, label}.

Usage:
  HF_TOKEN=... uv run etl/prepare.py --out data/generated
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset


def retail_conversations(token: str | None) -> list[dict]:
    """Group RetailBanking turns into {id, text (transcript), label (topic)}."""
    ds = load_dataset("oopere/RetailBanking-Conversations", split="train", token=token)
    turns: dict[str, list] = defaultdict(list)
    topic: dict[str, str] = {}
    for r in ds:
        cid = r["id_conversation"]
        turns[cid].append((r["sequence"], r["rol1"], r["rol2"]))
        topic[cid] = r["topic"]
    rows = []
    for cid, ts in turns.items():
        lines = []
        for _, customer, agent in sorted(ts):
            if customer:
                lines.append(f"Customer: {customer}")
            if agent:
                lines.append(f"Agent: {agent}")
        rows.append({"id": cid, "text": "\n".join(lines), "label": topic[cid]})
    return rows


def banking77(token: str | None) -> list[dict]:
    """Load banking77 as {id, text (query), label (intent name)}."""
    ds = load_dataset("legacy-datasets/banking77", split="test", token=token)
    names = ds.features["label"].names
    return [
        {"id": str(i), "text": r["text"], "label": names[r["label"]]}
        for i, r in enumerate(ds)
    ]


def write_jsonl(rows: list[dict], path: Path) -> None:
    """Write rows as one JSON object per line, and a sibling labels file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    labels = sorted({r["label"] for r in rows})
    path.with_suffix(".labels.json").write_text(json.dumps(labels, indent=2))
    print(f"{len(rows)} rows, {len(labels)} labels -> {path}")


def main() -> None:
    """Prepare both datasets into the output directory."""
    import os

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("data/generated"))
    a = p.parse_args()
    token = os.environ.get("HF_TOKEN")
    write_jsonl(retail_conversations(token), a.out / "conversations.jsonl")
    write_jsonl(banking77(token), a.out / "banking77.jsonl")


if __name__ == "__main__":
    main()
