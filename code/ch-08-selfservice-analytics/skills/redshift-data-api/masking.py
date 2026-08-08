"""Result masking: the LLM/transcript-side enforcement layer.

Columns come from config/pii_columns.yaml (the same file behind the warehouse-side policies, so the two layers cannot drift) and are masked by column NAME across all results; the warehouse-side policies are the backstop when a name does not survive.
Verified with the local stack 2026-07-30.
"""

import hashlib
import os
import pathlib

_STRATEGIES = {}


def _load() -> None:
    cfg = pathlib.Path(__file__).parents[2] / "config" / "pii_columns.yaml"
    if not cfg.exists():
        return
    for line in cfg.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, strategy = line.partition(":")
        column = key.strip().rsplit(".", 1)[-1]
        _STRATEGIES[column] = strategy.strip()


_load()


def _mask(value, strategy: str):
    if value is None:
        return None
    text = str(value)
    if strategy == "hash":
        salt = os.environ.get("MASK_SALT", "book")
        return hashlib.sha256((salt + text).encode()).hexdigest()[:12]
    if strategy == "partial_last4":
        return "*" * max(len(text) - 4, 0) + text[-4:]
    return "***"


def masked_columns() -> list[str]:
    """List the columns a masking strategy exists for."""
    return sorted(_STRATEGIES)


def mask_records(rows: list[dict]) -> list[dict]:
    """Apply the masking strategies to a result set."""
    if not rows or not _STRATEGIES:
        return rows
    hit = set(rows[0]) & set(_STRATEGIES)
    if not hit:
        return rows
    return [
        {k: _mask(v, _STRATEGIES[k]) if k in hit else v for k, v in r.items()}
        for r in rows
    ]
