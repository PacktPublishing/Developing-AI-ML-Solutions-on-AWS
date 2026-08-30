# /// script
# dependencies = ["fastapi", "uvicorn[standard]", "boto3", "ollama", "opensearch-py", "strands-agents"]
# ///
"""The underwriter app: one FastAPI service for the UI and the retrieval API.

GET / serves the ask/recommend page; POST /ask and POST /cases run the grounded
retrieval from retrieve.py and return {answer, sources}. POST /agent answers a
question that needs arithmetic as well as recall, by handing it to the Strands
agent with the memo search and the affordability tools. The same app runs under
uvicorn locally and, unchanged, in a Lambda container via the Lambda Web Adapter;
only the store endpoint and credentials differ.

Usage:
  STORE=opensearch BEDROCK_LOCAL=1 PYTHONPATH=src uv run app/main.py
"""

import os
from pathlib import Path

from agent import build_agent
from fastapi import FastAPI
from pydantic import BaseModel
from retrieve import ask_result, cases_result

app = FastAPI(title="Underwriting knowledge base")
STATIC = Path(__file__).parent / "static"


class AskBody(BaseModel):
    """A question for the knowledge base."""

    query: str
    k: int = 5


class CasesBody(BaseModel):
    """A new deal to compare against prior cases."""

    deal: str
    k: int = 5


class AgentBody(BaseModel):
    """A question that may need affordability arithmetic as well as retrieval."""

    query: str


class Source(BaseModel):
    """One cited loan."""

    loan_id: int
    borrower: str


class Answer(BaseModel):
    """A grounded answer plus the loans it cites."""

    answer: str
    sources: list[Source]


@app.post("/ask")
def ask(body: AskBody) -> Answer:
    """Answer a question, grounded in the nearest memos, with citations."""
    return Answer(**ask_result(body.query, body.k))


@app.post("/cases")
def cases(body: CasesBody) -> Answer:
    """Assemble the most similar prior cases into a draft recommendation."""
    return Answer(**cases_result(body.deal, body.k))


@app.post("/agent")
def agent(body: AgentBody) -> dict[str, str]:
    """Answer with the agent, which may call the memo search and the DTI tools."""
    return {"answer": str(build_agent()(body.query))}


@app.get("/healthz")
def healthz() -> dict[str, bool | str]:
    """Report which store the app is serving, for a quick liveness check."""
    return {"ok": True, "store": "opensearch"}


# -------------------------------------------------------------------------------
# Serve the frontend build: FastAPI checks the path operations above first and
# falls back to these files, so /ask and /cases win and everything else (a React
# build or this single page) is served with SPA fallback -- no extra glue.
# -------------------------------------------------------------------------------
app.frontend("/", directory=str(STATIC))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
