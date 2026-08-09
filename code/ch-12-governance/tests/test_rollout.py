"""The rollout: deterministic split, monotone widening, and router assignment."""

import copy
import json
from pathlib import Path

from appconfig import AppConfig, LocalFlagStore, evaluate_config
from router import Router

FLAGS = Path("local/flags/feature-flags.json")


def _variant(loan_id: str, cfg: dict) -> str:
    """Return the rollout variant a loan id resolves to."""
    return evaluate_config(cfg, {"loanId": loan_id})["challenger_rollout"]["_variant"]


def _with_pct(cfg: dict, pct: int) -> dict:
    """Return a copy of the config with the challenger split widened to pct."""
    out = copy.deepcopy(cfg)
    out["challenger_rollout"]["_variants"][0]["rule"] = (
        f'(split by:: $loanId pct::{pct} seed:: "rollout-2026")'
    )
    return out


def test_split_share_matches_the_flag():
    """About pct% of loans route to the challenger."""
    cfg = json.loads(FLAGS.read_text())
    loans = [f"L{i:06d}" for i in range(4000)]
    share = sum(_variant(x, cfg) == "challenger" for x in loans) / len(loans)
    assert 0.17 < share < 0.23  # ~20%


def test_assignment_is_deterministic():
    """A loan id always resolves to the same variant."""
    cfg = json.loads(FLAGS.read_text())
    assert len({_variant("L000042", cfg) for _ in range(10)}) == 1


def test_widening_never_flips_a_loan_back():
    """Bumping pct only adds loans to the challenger cohort; none leave it."""
    cfg = json.loads(FLAGS.read_text())
    loans = [f"L{i:06d}" for i in range(4000)]
    in20 = {x for x in loans if _variant(x, cfg) == "challenger"}
    in50 = {x for x in loans if _variant(x, _with_pct(cfg, 50)) == "challenger"}
    assert in20 < in50  # strictly grows, and the 20% cohort stays in
    assert in20 <= in50


def test_router_scores_with_the_picked_model():
    """The router routes each loan to the model the flag picks and decides on its pd."""
    appconfig = AppConfig(LocalFlagStore(FLAGS))
    calls = []

    def score(model_name, loan):
        calls.append((loan["loanId"], model_name))
        return 0.8 if model_name == "credit-challenger" else 0.1

    router = Router(appconfig, score)
    out = [router.route({"loanId": f"L{i:06d}"}) for i in range(200)]

    # every routed model matches the loan's variant, and the pd drives the decision
    for d in out:
        expected = (
            "credit-challenger" if d["variant"] == "challenger" else "credit-scorecard"
        )
        assert d["model"] == expected
        assert d["decision"] == ("refer" if d["pd"] >= 0.5 else "approve")
    assert {c[1] for c in calls} == {"credit-scorecard", "credit-challenger"}
