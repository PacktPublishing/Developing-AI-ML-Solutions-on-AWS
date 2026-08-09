# /// script
# dependencies = ["fastapi", "uvicorn[standard]", "boto3"]
# ///
"""The rollout gateway: one FastAPI service that flag-routes each application.

POST /score evaluates the challenger_rollout feature flag with the application's loanId,
routes it to the champion or challenger, and returns the decision plus which variant
served it. The flag comes from the get_appconfig seam -- a local flag store or real
AppConfig -- so widening the rollout is a flag edit with no redeploy. The same uvicorn
process runs locally and as a long-running Fargate service on AWS, where the from-source
AppConfig client polls the flag on its interval and keeps warm connections to the two
model endpoints.

Env: APPCONFIG_LOCAL, APPCONFIG_FLAGS, CHAMPION_URL, CHALLENGER_URL.

Usage:
  APPCONFIG_LOCAL=1 CHAMPION_URL=... CHALLENGER_URL=... uv run app/main.py
"""

import os

from fastapi import Depends, FastAPI
from pydantic import BaseModel, ConfigDict

from appconfig import get_appconfig
from models import http_scorer
from router import Router

app = FastAPI(title="Credit rollout gateway")
_router: Router | None = None


def get_router() -> Router:
    """Build the router once: the flag client plus the model-scoring seam."""
    global _router
    if _router is None:
        _router = Router(get_appconfig(), http_scorer())
    return _router


class Application(BaseModel):
    """An application: a loanId for the split, plus the model's feature fields."""

    model_config = ConfigDict(extra="allow")
    loanId: str


@app.post("/score")
def score(application: Application, router: Router = Depends(get_router)) -> dict:
    """Flag-route one application and return its decision and serving variant."""
    return router.route(application.model_dump())


@app.get("/healthz")
def healthz() -> dict:
    """Liveness check."""
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
