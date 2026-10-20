# /// script
# dependencies = ["boto3", "ollama", "opensearch-py"]
# ///
"""Answer questions over the memo knowledge base, grounded with citations.

Two modes: ask a question about past underwriting and credit policy, or assemble
the most similar prior cases into a draft recommendation the underwriter signs
off on. Both cite the source loan or policy so the underwriter can trace every
claim back to a document, and both can be narrowed by sector, exposure, and date.
The store is Amazon OpenSearch Service, a container locally and a domain on AWS.

Usage (from the chapter root):
  PYTHONPATH=src uv run src/retrieve.py ask --query "How is DTI assessed for grocery businesses?"
  PYTHONPATH=src uv run src/retrieve.py cases --deal "..." --sector Logistics --max-exposure 2000000
"""

import argparse

from models import embed, generate, get_runtime
from stores import get_store

# Headroom for the answer. Qwen3 is a reasoning model: it emits its thinking before
# the reply, and Ollama counts both against the budget, so too small a number
# returns an empty answer rather than a short one. Harmless on Bedrock, where the
# budget just caps the answer length.
MAX_TOKENS = 4000

# Policy passages always passed to the model, beside the k best results. Past
# memos far outnumber the policy, so without a reserved place a rule can be
# crowded out of the top k by cases that merely mention the same amounts.
POLICY_K = 2

ASK_SYSTEM = (
    "You are a credit underwriting assistant. Answer the question using only the"
    " passages provided, which are past memos and credit policy. If the answer is"
    " not in the passages, say you cannot find it in the knowledge base. After"
    " each claim, cite the source loan id or policy id in square brackets. Be"
    " concise and precise. Write in plain prose with no em dashes; use commas"
    " instead."
)

CASES_SYSTEM = (
    "You are a credit underwriting assistant. Using only the similar prior cases"
    " provided, draft a short recommendation for the new deal: note what the"
    " comparable borrowers looked like, what was approved, and the risks to"
    " watch. Cite each prior case by its loan id in square brackets. Do not make"
    " a final approve or decline decision; that is the underwriter's call. Write"
    " in plain prose with no em dashes."
)


def _ref(loan_id: int, borrower: str) -> str:
    """Return the citation tag: the loan id, or the policy id a policy is stored under."""
    return str(loan_id) if loan_id else borrower


def _context(hits: list[tuple[int, str, str, float]]) -> str:
    """Render retrieved chunks as citation-tagged context passages."""
    return "\n\n".join(f"[{_ref(lid, who)}] {content}" for lid, who, content, _ in hits)


def _sources_list(hits: list[tuple[int, str, str, float]]) -> list[dict]:
    """Return the cited loans and policies as [{loan_id, borrower}], in order.

    Deduplicated on the citation tag: every policy is stored under loan id 0, so
    the loan id alone would fold distinct policies into one source.
    """
    seen: dict[str, dict] = {}
    for loan_id, borrower, _, _ in hits:
        seen.setdefault(
            _ref(loan_id, borrower), {"loan_id": loan_id, "borrower": borrower}
        )
    return list(seen.values())


def _respond(
    system: str,
    header: str,
    body: str,
    text: str,
    k: int,
    empty: str,
    filters: dict | None = None,
) -> dict:
    """Retrieve, generate a grounded answer, and return {answer, sources}."""
    runtime = get_runtime()
    store = get_store()
    vector = embed(runtime, [text])[0]
    hits = store.search(runtime, text, k=k, filters=filters, vector=vector)
    rules = store.search(runtime, text, k=POLICY_K, sources=("policy",), vector=vector)
    hits += [h for h in rules if h[:3] not in {x[:3] for x in hits}]
    if not hits:
        return {"answer": empty, "sources": []}
    user = f"{header}: {text}\n\n{body}:\n{_context(hits)}"
    answer = generate(runtime, system, user, max_tokens=MAX_TOKENS)
    return {"answer": answer, "sources": _sources_list(hits)}


def ask_result(query: str, k: int = 5, filters: dict | None = None) -> dict:
    """Answer a question grounded in the k best chunks: {answer, sources}."""
    return _respond(
        ASK_SYSTEM,
        "Question",
        "Passages",
        query,
        k,
        "I cannot find anything relevant in the knowledge base.",
        filters,
    )


def cases_result(deal: str, k: int = 5, filters: dict | None = None) -> dict:
    """Assemble the k most similar prior cases into a recommendation: {answer, sources}."""
    return _respond(
        CASES_SYSTEM,
        "New deal",
        "Similar prior cases",
        deal,
        k,
        "I cannot find comparable cases in the knowledge base.",
        filters,
    )


def _format(result: dict) -> str:
    """Render a structured result as the answer followed by a Sources line."""
    if not result["sources"]:
        return result["answer"]
    cited = ", ".join(
        f"{s['loan_id']} {s['borrower']}" if s["loan_id"] else s["borrower"]
        for s in result["sources"]
    )
    return f"{result['answer']}\n\n> Sources: {cited}"


def ask(query: str, k: int = 5, filters: dict | None = None) -> str:
    """Answer a question grounded in the k best chunks, with citations."""
    return _format(ask_result(query, k, filters))


def cases(deal: str, k: int = 5, filters: dict | None = None) -> str:
    """Assemble the k most similar prior cases into a draft recommendation."""
    return _format(cases_result(deal, k, filters))


def main() -> None:
    """Parse the mode and query, then print the grounded answer."""
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)
    a = sub.add_parser("ask", help="answer a question with citations")
    a.add_argument("--query", required=True)
    c = sub.add_parser(
        "cases", help="assemble similar prior cases into a recommendation"
    )
    c.add_argument("--deal", required=True)
    for cmd in (a, c):
        cmd.add_argument("--k", type=int, default=5)
        cmd.add_argument("--sector")
        cmd.add_argument("--min-exposure", type=int)
        cmd.add_argument("--max-exposure", type=int)
        cmd.add_argument("--since", help="decision date, YYYY-MM-DD")
    args = p.parse_args()
    filters = {
        "sector": args.sector,
        "min_exposure": args.min_exposure,
        "max_exposure": args.max_exposure,
        "since": args.since,
    }
    if args.mode == "ask":
        print(ask(args.query, args.k, filters))
    else:
        print(cases(args.deal, args.k, filters))


if __name__ == "__main__":
    main()
