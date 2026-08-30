# /// script
# dependencies = ["boto3", "ollama", "opensearch-py", "strands-agents"]
# ///
"""The underwriting agent: the memo knowledge base plus affordability arithmetic.

Retrieval alone answers what was decided before. An underwriter also asks what
the numbers allow now, and that is arithmetic rather than recall, so it is a
tool the agent calls instead of something the model is asked to work out.

Usage (from the chapter root):
  PYTHONPATH=src uv run src/agent.py --ask \
      "Customer earns 90,000 a month with 12,000 of existing repayments. \
       What can we lend over 12 months at 18%, and have we done similar before?"
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
    "You are a credit underwriting assistant. You have two kinds of help: past"
    " memos, which you must cite by loan id in square brackets, and an"
    " affordability calculator. Never compute a debt-to-income ratio or an"
    " instalment yourself, always call the tool, because the underwriter checks"
    " these numbers. The debt-to-income ceiling is credit policy and comes from"
    " configuration, so never choose one yourself and always report the ceiling"
    " the tool returns. Recommend an amount and say what it assumes. Do not make the"
    " final approve or decline decision, that is the underwriter's call. Write in"
    " plain prose with no em dashes, use commas instead."
)


@tool
def search_memos(query: str, k: int = 5) -> str:
    """Search past underwriting memos and return passages tagged by loan id."""
    hits = get_store().search(get_runtime(), query, k=k)
    return _context(hits)


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
        tools=[search_memos, assess_affordability, recommend_amount],
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
