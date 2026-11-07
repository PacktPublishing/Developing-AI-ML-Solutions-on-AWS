# /// script
# dependencies = ["boto3", "ollama", "opensearch-py", "strands-agents"]
# ///
"""The underwriting agent: the memo knowledge base plus affordability arithmetic.

Retrieval alone answers what was decided before. An underwriter also asks what
the numbers allow now, and that is arithmetic rather than recall, so it is a
tool the agent calls instead of something the model is asked to work out.

Usage (from the chapter root):
  PYTHONPATH=src uv run src/agent.py --ask \
      "Review CASE-20260000 for a 12 month facility at 18 percent."
"""

import argparse
import os

from affordability import CEILING_PERCENT, dti, max_affordable_amount
from models import BEDROCK_REGION, OLLAMA_TEXT_MODEL, TEXT_MODEL, get_runtime
from retrieve import _context
from stores import get_store
from strands import Agent, tool
from strands.models import BedrockModel

SYSTEM = (
    "You are a credit underwriting assistant. You have four kinds of help: the"
    " applicant's uploaded bank statement, any supporting documents the"
    " underwriter added to the case, which you name when you use them, past"
    " memos, which you must cite by loan id in square brackets, and an"
    " affordability calculator. Never compute a debt-to-income ratio or an"
    " instalment yourself, always call the tool, because the underwriter checks"
    " these numbers. When the underwriter names a case reference, read the"
    " statement first and use the income and existing repayments it returns"
    " rather than any figure quoted in the question. The income it returns is"
    " already the income credit policy allows: it excludes own-account"
    " transfers, asset sales and loan disbursements, so use it as it stands and"
    " say what was excluded when the underwriter asks why it is lower than the"
    " credits. If the statement reconciliation is not reconciled, say so first,"
    " list the rows that break the running balance, and treat the figures as"
    " unconfirmed; not_checked means nothing was verified, not that it passed."
    " A figure in a supporting document is what that document claims, so never"
    " put one into the affordability tools in place of the statement's."
    " The debt-to-income ceiling is credit policy and comes from"
    " configuration, so never choose one yourself and always report the ceiling"
    " the tool returns. Recommend an amount and say what it assumes. Do not make the"
    " final approve or decline decision, that is the underwriter's call. Write in"
    " plain prose with no em dashes, use commas instead."
)


@tool
def search_memos(query: str, k: int = 5) -> str:
    """Search past memos and credit policy, passages tagged by loan or policy id.

    Only decided cases and policy are searched, never the document under review,
    so a submission cannot come back as its own precedent.
    """
    hits = get_store().search(get_runtime(), query, k=k)
    return _context(hits)


# -------------------------------------------------------------------------------
# Tool to read the applicant's uploaded bank statement
# -------------------------------------------------------------------------------
@tool
def read_case(case_id: str) -> dict:
    """Return the figures parsed from the applicant's uploaded bank statement.

    The income and existing repayments come from totalling the statement at
    upload, not from reading the text, so the affordability tools are given
    figures rather than an impression of them.

    `monthly_income` is the income the policy allows a ratio to use, which is
    not the total of the credits: own-account transfers, asset sales and loan
    disbursements are listed in `excluded_inflows` and left out. `reconciliation`
    is reconciled, breaks, or not_checked, and not_checked is not a pass.
    The newest upload answers, and `version` says which one that was.
    """
    doc = get_store().read_submission(case_id)
    if not doc:
        return {"error": f"no statement uploaded for {case_id}"}
    return {
        "case_id": doc["case_id"],
        "version": doc["version"],
        "monthly_income": doc["monthly_income"],
        "total_inflows": doc["total_inflows"],
        "excluded_inflows": doc["excluded_inflows"],
        "existing_repayments": doc["existing_repayments"],
        "months_covered": doc["months"],
        "reconciliation": doc["reconciliation"],
        "balance_breaks": doc["balance_breaks"],
    }


# -------------------------------------------------------------------------------
# Tool to read the supporting documents the underwriter added to the case
# -------------------------------------------------------------------------------
@tool
def read_context(case_id: str) -> dict:
    """Return the supporting documents an underwriter added to this case.

    A tax filing, an invoice, a contract, a page of management accounts: what
    the applicant sent besides the bank statement. Each was read by the document
    reader at upload, and each carries what that reading found and the document's
    own text. The figures in them are the document's claims, not verified
    income: the affordability arithmetic still comes from read_case. Name the
    document when you use something from it, so the underwriter can check it.
    """
    documents = get_store().list_submissions(case_id, source="context")
    if not documents:
        return {"documents": [], "note": f"no supporting documents for {case_id}"}
    return {
        "documents": [
            {
                "doc_name": doc["doc_name"],
                "doc_type": doc["doc_type"],
                "uploaded_at": doc["uploaded_at"],
                # enough of the document to answer from, not the whole file
                "text": doc["text"][:4000],
            }
            for doc in documents
        ]
    }


# -------------------------------------------------------------------------------
# Tool to assess the affordability of one proposed loan
# -------------------------------------------------------------------------------
@tool
def assess_affordability(
    monthly_income: float,
    existing_repayments: float,
    amount: float,
    annual_rate_percent: float,
    tenor_months: int,
) -> dict:
    """Return the instalment, DTI percent, and headroom for one proposed loan.

    The DTI ceiling is credit policy and is not an argument: it comes from
    configuration, so the answer cannot drift onto a limit nobody approved.
    """
    result = dti(
        monthly_income,
        existing_repayments,
        amount,
        annual_rate_percent,
        tenor_months,
        CEILING_PERCENT,
    )
    return {
        "instalment": result.instalment,
        "dti_percent": result.dti_percent,
        "within_ceiling": result.within_ceiling,
        "headroom": result.headroom,
        "ceiling_percent": CEILING_PERCENT,
    }


# -------------------------------------------------------------------------------
# Tool to recommend the largest loan amount that keeps DTI within the ceiling
# -------------------------------------------------------------------------------
@tool
def recommend_amount(
    monthly_income: float,
    existing_repayments: float,
    annual_rate_percent: float,
    tenor_months: int,
) -> dict:
    """Return the largest loan amount that keeps DTI within the policy ceiling."""
    amount = max_affordable_amount(
        monthly_income,
        existing_repayments,
        annual_rate_percent,
        tenor_months,
        CEILING_PERCENT,
    )
    return {
        "max_amount": amount,
        "tenor_months": tenor_months,
        "annual_rate_percent": annual_rate_percent,
        "ceiling_percent": CEILING_PERCENT,
    }


def build_model():
    """Return the Strands model: Bedrock, or Ollama when BEDROCK_LOCAL is set."""
    if os.environ.get("BEDROCK_LOCAL") == "1":
        from strands.models.ollama import OllamaModel

        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        return OllamaModel(host=host, model_id=OLLAMA_TEXT_MODEL)
    return BedrockModel(
        model_id=TEXT_MODEL,
        region_name=BEDROCK_REGION,
        streaming=False,
        temperature=0.0,
    )


def build_agent() -> Agent:
    """Build the agent with the memo search and the two affordability tools."""
    return Agent(
        model=build_model(),
        tools=[
            search_memos,
            read_case,
            read_context,
            assess_affordability,
            recommend_amount,
        ],
        system_prompt=SYSTEM,
        callback_handler=None,
    )


def main() -> None:
    """Answer one underwriting question with retrieval and affordability tools."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ask", required=True, help="the underwriter's question")
    args = p.parse_args()
    print(build_agent()(args.ask))


if __name__ == "__main__":
    main()
