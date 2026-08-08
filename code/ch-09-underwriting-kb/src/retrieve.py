# /// script
# dependencies = ["boto3", "psycopg2-binary", "ollama", "opensearch-py"]
# ///
"""Answer questions over the memo knowledge base, grounded with citations.

Two modes: ask a question about past underwriting, or assemble the most similar
prior cases into a draft recommendation the underwriter signs off on. Both cite
the source loan so the underwriter can trace every claim back to a memo. The
vector store is chosen by STORE (pgvector by default, or opensearch).

Usage (from the chapter root):
  PYTHONPATH=src uv run src/retrieve.py ask --query "How is DTI assessed for grocery businesses?"
  STORE=opensearch PYTHONPATH=src uv run src/retrieve.py ask --query "..."
"""

import argparse

from models import generate, get_runtime
from stores import get_store

# Headroom for the answer: a small local model (Qwen3 0.6b) spends much of its
# budget on reasoning over the retrieved passages, so leave room for the reply.
# Harmless on Bedrock, where the budget just caps the answer length.
MAX_TOKENS = 1500

ASK_SYSTEM = (
    "You are a credit underwriting assistant. Answer the question using only the"
    " memo passages provided. If the answer is not in the passages, say you"
    " cannot find it in the knowledge base. After each claim, cite the source"
    " loan id in square brackets. Be concise and precise. Write in plain prose"
    " with no em dashes; use commas instead."
)

CASES_SYSTEM = (
    "You are a credit underwriting assistant. Using only the similar prior cases"
    " provided, draft a short recommendation for the new deal: note what the"
    " comparable borrowers looked like, what was approved, and the risks to"
    " watch. Cite each prior case by its loan id in square brackets. Do not make"
    " a final approve or decline decision; that is the underwriter's call. Write"
    " in plain prose with no em dashes."
)


def _context(hits: list[tuple[int, str, str, float]]) -> str:
    """Render retrieved memo chunks as citation-tagged context passages."""
    return "\n\n".join(f"[{loan_id}] {content}" for loan_id, _, content, _ in hits)


def _sources_list(hits: list[tuple[int, str, str, float]]) -> list[dict]:
    """Return the cited loans as [{loan_id, borrower}], deduplicated, in order."""
    seen: dict[int, str] = {}
    for loan_id, borrower, _, _ in hits:
        seen.setdefault(loan_id, borrower)
    return [{"loan_id": lid, "borrower": name} for lid, name in seen.items()]


def _respond(
    system: str, header: str, body: str, text: str, k: int, empty: str
) -> dict:
    """Retrieve, generate a grounded answer, and return {answer, sources}."""
    runtime = get_runtime()
    hits = get_store().search(runtime, text, k=k)
    if not hits:
        return {"answer": empty, "sources": []}
    user = f"{header}: {text}\n\n{body}:\n{_context(hits)}"
    answer = generate(runtime, system, user, max_tokens=MAX_TOKENS)
    return {"answer": answer, "sources": _sources_list(hits)}


def ask_result(query: str, k: int = 5) -> dict:
    """Answer a question grounded in the k nearest memo chunks: {answer, sources}."""
    return _respond(
        ASK_SYSTEM,
        "Question",
        "Memo passages",
        query,
        k,
        "I cannot find anything relevant in the knowledge base.",
    )


def cases_result(deal: str, k: int = 5) -> dict:
    """Assemble the k most similar prior cases into a recommendation: {answer, sources}."""
    return _respond(
        CASES_SYSTEM,
        "New deal",
        "Similar prior cases",
        deal,
        k,
        "I cannot find comparable cases in the knowledge base.",
    )


def _format(result: dict) -> str:
    """Render a structured result as the answer followed by a Sources line."""
    if not result["sources"]:
        return result["answer"]
    cited = ", ".join(f"{s['loan_id']} {s['borrower']}" for s in result["sources"])
    return f"{result['answer']}\n\n> Sources: {cited}"


def ask(query: str, k: int = 5) -> str:
    """Answer a question grounded in the k nearest memo chunks, with citations."""
    return _format(ask_result(query, k))


def cases(deal: str, k: int = 5) -> str:
    """Assemble the k most similar prior cases into a draft recommendation."""
    return _format(cases_result(deal, k))


def main() -> None:
    """Parse the mode and query, then print the grounded answer."""
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)
    a = sub.add_parser("ask", help="answer a question with citations")
    a.add_argument("--query", required=True)
    a.add_argument("--k", type=int, default=5)
    c = sub.add_parser(
        "cases", help="assemble similar prior cases into a recommendation"
    )
    c.add_argument("--deal", required=True)
    c.add_argument("--k", type=int, default=5)
    args = p.parse_args()
    if args.mode == "ask":
        print(ask(args.query, args.k))
    else:
        print(cases(args.deal, args.k))


if __name__ == "__main__":
    main()
