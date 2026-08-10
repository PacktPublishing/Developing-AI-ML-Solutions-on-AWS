# /// script
# dependencies = ["fastapi", "uvicorn[standard]", "boto3"]
# ///
"""The rollout gateway: one FastAPI service that flag-routes each application.

POST /score evaluates the challenger_rollout feature flag with the application's loanId,
routes it to the champion or challenger, and returns the decision plus which variant
served it. The flag comes from the AWS AppConfig agent sidecar, which evaluates the
rules -- local development mode over an Ion file locally, the deployed configuration
on ECS -- so widening the rollout is a flag edit with no redeploy. The same uvicorn
process runs locally and as a long-running Fargate service on AWS.

Env: APPCONFIG_AGENT_URL (+ APPCONFIG_APP / APPCONFIG_ENV / APPCONFIG_PROFILE),
     CHAMPION_URL, CHALLENGER_URL.

Usage:
  CHAMPION_URL=... CHALLENGER_URL=... uv run app/main.py
"""

import os
from typing import Annotated

from appconfig import get_appconfig
from fastapi import Depends, FastAPI
from models import get_scorer
from pydantic import BaseModel, ConfigDict
from router import Router

app = FastAPI(title="Credit rollout gateway")
_router: Router | None = None


def get_router() -> Router:
    """Build the router once: the flag client plus the model-scoring seam."""
    global _router
    if _router is None:
        _router = Router(get_appconfig(), get_scorer())
    return _router


class Application(BaseModel):
    """An application: a loanId for the split, plus the model's feature fields."""

    model_config = ConfigDict(extra="allow")
    loanId: str


class Decision(BaseModel):
    """The scoring response: the decision and which variant and model served it."""

    loanId: str
    variant: str
    model: str
    pd: float
    decision: str


@app.post("/score")
def score(
    application: Application, router: Annotated[Router, Depends(get_router)]
) -> Decision:
    """Flag-route one application and return its decision and serving variant."""
    return Decision(**router.route(application.model_dump()))


@app.get("/healthz")
def healthz() -> dict[str, bool]:
    """Liveness check."""
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
