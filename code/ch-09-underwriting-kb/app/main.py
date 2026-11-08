# /// script
# dependencies = ["fastapi", "uvicorn[standard]", "python-multipart", "pdfplumber", "boto3", "botocore[crt]", "ollama", "opensearch-py", "strands-agents", "pyjwt[crypto]"]
# ///
"""Serve underwriting queries, statement uploads and supporting documents.

The FastAPI service runs locally under uvicorn and in Lambda through the Lambda
Web Adapter. Query routes call retrieval or agent functions; upload routes index
case documents. Authentication uses the configured Cognito user pool.

Usage:
  STORE=opensearch BEDROCK_LOCAL=1 PYTHONPATH=src uv run app/main.py
"""

import os
import tempfile
from pathlib import Path
from typing import Annotated

from agent import build_agent
from audit import record, underwriter
from documents import Unsupported
from documents import submit as read_context_document
from fastapi import Depends, FastAPI, Form, Header, HTTPException, UploadFile
from models import get_runtime
from pydantic import BaseModel
from retrieve import ask_result, cases_result
from statements import Unreadable, submit
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
    """Parse, validate and index a new statement version; return 422 on invalid input."""
    suffix = Path(file.filename or "statement.pdf").suffix or ".pdf"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        summary = submit(tmp_path, get_store(), get_runtime())
    except Unreadable as err:
        # nothing was indexed, so the applicant can send a readable copy and the
        # case is where it was, rather than holding a statement of no income
        raise HTTPException(status_code=422, detail=str(err)) from err
    finally:
        os.unlink(tmp_path)
    record(who, "upload", summary.case_id, [])
    return {
        "case_id": summary.case_id,
        "version": summary.version,
        "holder": summary.holder,
        "monthly_income": summary.monthly_income,
        "total_inflows": summary.total_inflows,
        "excluded_inflows": summary.excluded_inflows,
        "existing_repayments": summary.existing_repayments,
        "months_covered": summary.months,
        "transactions": summary.transactions,
        "reconciliation": summary.reconciliation,
        "balance_breaks": summary.balance_breaks,
    }


@app.post("/context")
async def add_context(
    case_id: Annotated[str, Form()], file: UploadFile, who: Underwriter
) -> dict:
    """Read and index a supporting case document, retaining previous documents."""
    suffix = Path(file.filename or "document.pdf").suffix or ".pdf"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        reading = read_context_document(tmp_path, case_id, get_store(), get_runtime())
    except Unsupported as err:
        raise HTTPException(status_code=422, detail=str(err)) from err
    finally:
        os.unlink(tmp_path)
    record(who, "context", f"{case_id} {reading.doc_name}", [])
    return {
        "case_id": case_id,
        "doc_name": reading.doc_name,
        "doc_type": reading.doc_type,
        "summary": reading.summary,
        "figures": reading.figures,
        "dates": reading.dates,
        "concerns": reading.concerns,
        "pages_with_text": sum(1 for page in reading.pages if page.strip()),
    }


@app.get("/context/{case_id}")
def list_context(case_id: str, who: Underwriter) -> dict:
    """List the supporting documents held for a case."""
    documents = get_store().list_submissions(case_id, source="context")
    return {
        "case_id": case_id,
        "documents": [
            {
                "doc_name": doc["doc_name"],
                "doc_type": doc["doc_type"],
                "uploaded_at": doc["uploaded_at"],
            }
            for doc in documents
        ],
    }


@app.delete("/context/{case_id}")
def drop_context(case_id: str, who: Underwriter) -> dict[str, int]:
    """Remove the supporting documents held for a case."""
    record(who, "drop-context", case_id, [])
    return {"deleted": get_store().drop_submission(case_id, source="context")}


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
