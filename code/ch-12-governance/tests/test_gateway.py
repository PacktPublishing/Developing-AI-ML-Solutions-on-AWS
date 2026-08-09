"""The FastAPI rollout gateway: flag-routed scoring over HTTP."""

from pathlib import Path

from fastapi.testclient import TestClient

from appconfig import AppConfig, LocalFlagStore
from main import app, get_router
from router import Router

FLAGS = Path("local/flags/feature-flags.json")


def _client(score):
    """Return a TestClient whose router uses the local flags and a stub scorer."""
    app.dependency_overrides[get_router] = lambda: Router(
        AppConfig(LocalFlagStore(FLAGS)), score
    )
    return TestClient(app)


def test_score_routes_by_the_flag_and_decides():
    """POST /score returns the serving variant, its model, and the decision."""
    client = _client(lambda m, loan: 0.8 if m == "credit-challenger" else 0.1)
    seen = set()
    for i in range(200):
        body = client.post(
            "/score", json={"loanId": f"L{i:06d}", "annual_income": 50000}
        ).json()
        expected = (
            "credit-challenger"
            if body["variant"] == "challenger"
            else "credit-scorecard"
        )
        assert body["model"] == expected
        assert body["decision"] == ("refer" if body["pd"] >= 0.5 else "approve")
        seen.add(body["variant"])
    assert seen == {"champion", "challenger"}  # both variants served
    app.dependency_overrides.clear()


def test_healthz():
    """The liveness probe answers ok."""
    assert TestClient(app).get("/healthz").json() == {"ok": True}
