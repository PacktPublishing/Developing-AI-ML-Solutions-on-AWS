# /// script
# dependencies = ["fastapi", "uvicorn[standard]", "boto3", "botocore[crt]"]
# ///
"""Serve application scoring through a FastAPI gateway.

POST /score evaluates challenger_rollout with the application's loanId, calls
the selected model and returns its probability, decision, variant and model
name. The AppConfig agent evaluates the flag. The gateway runs under uvicorn
locally and on Fargate.

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
    """Create and cache the feature-flag client and model router."""
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
    """Score an application with the selected model and return the decision."""
    return Decision(**router.route(application.model_dump()))


@app.get("/healthz")
def healthz() -> dict[str, bool]:
    """Liveness check."""
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
