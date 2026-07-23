# /// script
# dependencies = ["boto3", "psycopg2-binary", "ollama", "strands-agents", "bedrock-agentcore"]
# ///
"""The underwriting assistant: a grounded query mode and an agent mode.

Two ways to use the knowledge base. The ask mode answers one question with
retrieval-augmented generation and a guardrail, the direct path a person takes.
The decide mode runs a Strands agent that works a whole deal: it looks up the
sector and policy, assigns an internal risk profile because there is no external
score, and routes the decision to the role that may approve it. Both run on
Bedrock, or on Ollama with BEDROCK_LOCAL=1. The serve mode hosts the agent on
the Amazon Bedrock AgentCore runtime contract.

Usage (from the chapter root):
  PYTHONPATH=. uv run underwriting-agent/agent.py ask --query "What DSCR floor applies to a solar SPV?"
  PYTHONPATH=. uv run underwriting-agent/agent.py decide \
      --deal "GreenRoof Installers, residential solar, seeks a 3.5 million term loan"
  PYTHONPATH=. uv run underwriting-agent/agent.py serve
"""

import argparse
import json
import os
import pathlib
import re

from strands import Agent, tool
from strands.models import BedrockModel

from authority import decide, load_matrix
from guardrails import guarded_generate
from guardrails import load as load_guardrail
from models import BEDROCK_REGION, OLLAMA_TEXT_MODEL, TEXT_MODEL, generate, get_runtime
from stores import search

# -------------------------------------------------------------------------------
# System prompts
# -------------------------------------------------------------------------------
ASK_SYSTEM = (
    "You are a credit underwriting assistant for a lender to the solar sector."
    " Answer the question using only the context passages provided. If the answer"
    " is not in the context, say you cannot find it in the knowledge base. After"
    " each claim, cite the source document id in square brackets. Be concise and"
    " precise. Write in plain prose with no em dashes; use commas instead."
)

AGENT_SYSTEM = (
    "You are a credit underwriting assistant for a lender to the solar sector."
    " These borrowers have no external credit score, so you support the"
    " underwriter's judgment, you do not replace it. Work a deal in this order."
    " First call lookup_sector to retrieve the relevant sector profile, the"
    " five-factor rating scorecard, and the credit policy. Using only that"
    " retrieved guidance, grade the five factors (financial strength; market and"
    " sector position; sponsor and execution; security package; climate and"
    " transition) and map the result to an internal risk profile from 1 to 10,"
    " where 10 is the most risky. Work from the deal description and the sector"
    " baseline. If a specific figure is not given, assume the sector norm, say so"
    " in one line, and proceed; do not ask the user for more information. Then"
    " call route_decision with the deal exposure in US dollars and your risk"
    " profile to find which role may approve it. Use the exact role name that"
    " route_decision returns; do not rename, paraphrase, or invent a role."
    " Finish with a short recommendation that states the risk profile,"
    " the deciding role, whether the decision escalates, and the key risks to"
    " watch, citing the sector and policy document ids exactly as they appear."
    " Do not make a final"
    " approve or decline decision; that is the approver's call. Write in plain"
    " prose with no em dashes."
)


# -------------------------------------------------------------------------------
# Agent tools
# -------------------------------------------------------------------------------
@tool
def lookup_sector(query: str) -> str:
    """Search the credit knowledge base for the sector profile and policy for a deal."""
    hits = search(get_runtime(), query, k=5)
    return "\n\n".join(f"[{doc_id}] {content}" for doc_id, content, _ in hits)


@tool
def route_decision(exposure_usd: float, risk_profile: int) -> dict:
    """Return which role may approve a deal, given its exposure in USD and risk profile."""
    return decide(exposure_usd, risk_profile, load_matrix())


# -------------------------------------------------------------------------------
# Model and agent construction
# -------------------------------------------------------------------------------
def build_model():
    """Return the Strands model: Bedrock, or Ollama when BEDROCK_LOCAL is set.

    The guardrail is applied on the ask path, not here. A guardrail wrapped
    around an agent inspects every tool exchange, and the prompt-attack filter in
    particular tends to flag the agent's own instructions and tool results, so we
    keep the guardrail on the single-shot grounded answer where it belongs.
    """
    if os.environ.get("BEDROCK_LOCAL") == "1":
        from strands.models.ollama import OllamaModel

        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        return OllamaModel(host=host, model_id=OLLAMA_TEXT_MODEL)
    # non-streaming keeps tool use reliable for Qwen3; temperature 0 keeps the
    # agent faithful to the tool output instead of inventing role names
    return BedrockModel(
        model_id=TEXT_MODEL,
        region_name=BEDROCK_REGION,
        streaming=False,
        temperature=0.0,
    )


def build_agent() -> Agent:
    """Build the Strands agent with the sector-lookup and routing tools."""
    # callback_handler=None turns off Strands' streaming print so we emit the
    # final recommendation once, from run_decide.
    return Agent(
        model=build_model(),
        tools=[lookup_sector, route_decision],
        system_prompt=AGENT_SYSTEM,
        callback_handler=None,
    )


# -------------------------------------------------------------------------------
# Ask, decide, and serve modes
# -------------------------------------------------------------------------------
def run_ask(query: str, k: int, use_guardrail: bool, capture: bool) -> None:
    """Answer one question with grounded retrieval and print the sources."""
    runtime = get_runtime()
    hits = search(runtime, query, k)
    context = "\n\n".join(f"[{doc_id}] {content}" for doc_id, content, _ in hits)
    if use_guardrail:
        answer = guarded_generate(runtime, ASK_SYSTEM, query, context, load_guardrail())
    else:
        answer = generate(
            runtime, ASK_SYSTEM, f"Question: {query}\n\nContext passages:\n{context}"
        )

    print(answer)
    print("\nSources:")
    for doc_id, _, similarity in hits:
        print(f"  {doc_id}  (similarity {similarity:.3f})")

    if capture:
        out_dir = pathlib.Path("fixtures/expected")
        out_dir.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^a-z0-9]+", "-", query.lower()).strip("-")[:60]
        local = os.environ.get("BEDROCK_LOCAL") == "1"
        provider = "local" if local else "bedrock"
        # record the model too: a capture that says only "local" cannot be
        # compared against anything later, since the answer depends on which
        # Ollama model produced it
        record = {
            "query": query,
            "k": k,
            "provider": provider,
            "model": OLLAMA_TEXT_MODEL if local else TEXT_MODEL,
            "retrieved": [doc_id for doc_id, _, _ in hits],
            "answer": answer,
        }
        (out_dir / f"{slug}.{provider}.json").write_text(json.dumps(record, indent=2))
        print(f"\nCaptured expected output to fixtures/expected/{slug}.{provider}.json")


def run_decide(deal: str) -> None:
    """Run the agent on a whole deal and print the recommendation."""
    result = build_agent()(deal)
    print(result)


def run_serve() -> None:
    """Host the agent on the AgentCore runtime contract."""
    from bedrock_agentcore.runtime import BedrockAgentCoreApp

    app = BedrockAgentCoreApp()

    @app.entrypoint
    def invoke(payload: dict) -> str:
        return str(build_agent()(payload.get("prompt", "")))

    app.run(port=int(os.environ.get("PORT", "8080")))


# -------------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------------
def main() -> None:
    """Parse arguments and dispatch to the ask, decide, or serve mode."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="one grounded, cited answer")
    ask.add_argument("--query", required=True)
    ask.add_argument("--k", type=int, default=5)
    ask.add_argument("--guardrail", action="store_true")
    ask.add_argument("--capture", action="store_true")

    dec = sub.add_parser("decide", help="run the agent on a whole deal")
    dec.add_argument("--deal", required=True)

    sub.add_parser("serve", help="host the agent on the AgentCore runtime contract")

    args = parser.parse_args()
    if args.command == "ask":
        run_ask(args.query, args.k, args.guardrail, args.capture)
    elif args.command == "decide":
        run_decide(args.deal)
    elif args.command == "serve":
        run_serve()


if __name__ == "__main__":
    main()
