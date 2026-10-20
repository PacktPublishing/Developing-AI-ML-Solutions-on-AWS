# /// script
# dependencies = ["fastapi", "uvicorn[standard]", "python-multipart", "pdfplumber", "boto3", "ollama", "opensearch-py", "strands-agents", "pyjwt[crypto]"]
# ///
"""The underwriter app: one FastAPI service for the UI and the retrieval API.

GET / serves the ask/recommend page; POST /upload takes an applicant's bank
statement, totals it, and indexes it as the document under review; POST /ask and
POST /cases run the grounded
retrieval from retrieve.py and return {answer, sources}. POST /agent answers a
question that needs arithmetic as well as recall, by handing it to the Strands
agent with the memo search and the affordability tools. The same app runs under
uvicorn locally and, unchanged, in a Lambda container via the Lambda Web Adapter;
only the store endpoint and credentials differ.

Every route but the page and its sign-in settings needs a token from the
underwriters' Cognito user pool, and every answer is written to the audit log
with the underwriter who asked. On AWS, API Gateway checks the token first and
the app checks it again; locally the pool is cognito-shim and only the app checks.

Usage:
  STORE=opensearch BEDROCK_LOCAL=1 PYTHONPATH=src uv run app/main.py
"""

import os
import tempfile
from pathlib import Path
from typing import Annotated

from agent import build_agent
from audit import record, underwriter
from fastapi import Depends, FastAPI, Header, HTTPException, UploadFile
from models import get_runtime
from pydantic import BaseModel
from retrieve import ask_result, cases_result
from statements import submit
from stores import get_store

app = FastAPI(title="Underwriting knowledge base")
STATIC = Path(__file__).parent / "static"


class Scope(BaseModel):
    """Optional filters that narrow the precedent; policy always stays in scope."""

    sector: str | None = None
    min_exposure: int | None = None
    max_exposure: int | None = None
    since: str | None = None  # decision date, YYYY-MM-DD

    def filters(self) -> dict:
        """Return the filters as the store's search expects them."""
        return self.model_dump(
            include={"sector", "min_exposure", "max_exposure", "since"}
        )


class AskBody(Scope):
    """A question for the knowledge base."""

    query: str
    k: int = 5


class CasesBody(Scope):
    """A new deal to compare against prior cases."""

    deal: str
    k: int = 5


class AgentBody(BaseModel):
    """A question that may need affordability arithmetic as well as retrieval."""

    query: str


class Source(BaseModel):
    """One cited loan, or a policy (loan_id 0, borrower holds the policy id)."""

    loan_id: int
    borrower: str


class Answer(BaseModel):
    """A grounded answer plus the loans and policies it cites."""

    answer: str
    sources: list[Source]


def signed_in(authorization: Annotated[str | None, Header()] = None) -> str:
    """Return the underwriter behind the request, or answer 401 before any work."""
    try:
        return underwriter(authorization)
    except PermissionError as err:
        raise HTTPException(status_code=401, detail="Unauthorized") from err


Underwriter = Annotated[str, Depends(signed_in)]


def _cited(result: dict) -> list:
    """Return the loan or policy ids an answer cited, for the audit line."""
    return [s["loan_id"] or s["borrower"] for s in result["sources"]]


@app.post("/ask")
def ask(body: AskBody, who: Underwriter) -> Answer:
    """Answer a question, grounded in memos and policy, with citations."""
    result = ask_result(body.query, body.k, body.filters())
    record(who, "ask", body.query, _cited(result), body.filters())
    return Answer(**result)


@app.post("/cases")
def cases(body: CasesBody, who: Underwriter) -> Answer:
    """Assemble the most similar prior cases into a draft recommendation."""
    result = cases_result(body.deal, body.k, body.filters())
    record(who, "cases", body.deal, _cited(result), body.filters())
    return Answer(**result)


@app.post("/agent")
def agent(body: AgentBody, who: Underwriter) -> dict[str, str]:
    """Answer with the agent, which may call the memo search and the DTI tools."""
    answer = str(build_agent()(body.query))
    record(who, "agent", body.query, [])
    return {"answer": answer}


@app.post("/upload")
async def upload(file: UploadFile, who: Underwriter) -> dict:
    """Take an applicant's bank statement, total it, and index it for review.

    The figures are parsed here, once, and stored with the document, so the
    agent later reads them back from OpenSearch instead of adding up a table.
    """
    suffix = Path(file.filename or "statement.pdf").suffix or ".pdf"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        summary = submit(tmp_path, get_store(), get_runtime())
    finally:
        os.unlink(tmp_path)
    record(who, "upload", summary.case_id, [])
    return {
        "case_id": summary.case_id,
        "holder": summary.holder,
        "monthly_income": summary.monthly_income,
        "existing_repayments": summary.existing_repayments,
        "months_covered": summary.months,
        "transactions": summary.transactions,
        "balance_breaks": summary.balance_breaks,
    }


@app.delete("/upload/{case_id}")
def drop(case_id: str, who: Underwriter) -> dict[str, int]:
    """Remove an uploaded statement once its review is done."""
    record(who, "drop", case_id, [])
    return {"deleted": get_store().drop_submission(case_id)}


@app.get("/auth-config")
def auth_config() -> dict[str, str]:
    """Tell the page where to sign in; empty when the app runs open."""
    client_id = os.environ.get("COGNITO_CLIENT_ID", "")
    if not client_id:
        return {"client_id": "", "endpoint": ""}
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    endpoint = os.environ.get(
        "COGNITO_ENDPOINT", f"https://cognito-idp.{region}.amazonaws.com/"
    )
    return {"client_id": client_id, "endpoint": endpoint}


@app.get("/healthz", dependencies=[Depends(signed_in)])
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
