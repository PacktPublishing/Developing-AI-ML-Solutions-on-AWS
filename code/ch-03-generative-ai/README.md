# Generative AI on AWS: the underwriting assistant

The running example for the generative AI chapter. The lender from the data
chapter has moved into corporate lending to the solar sector: installers,
developers, and the project companies that own the assets. The amounts are
larger and the borrowers are young companies in a new market, so there is no
external credit score. An underwriter reads the documents, compares the deal
against the sector, assigns an internal risk profile, and routes the decision to
whoever may approve it. Most of the time goes on reading and comparing, so we
build an assistant that helps with both and shortens time to decision.

The chapter builds it in four moves, one per topic:

- embeddings turn the knowledge base into vectors
- retrieval-augmented generation drafts grounded answers, with a guardrail
- an agent works a whole deal and routes it to the right approver
- the whole loop runs locally first, with the same code that runs on AWS

The vector store is a real engine in Docker, no mocks: Postgres with pgvector. On
AWS this is Aurora PostgreSQL Serverless (express configuration) with the same extension, and the retrieval
SQL does not change.

Amazon Bedrock is the one cloud dependency. The code always calls the
`bedrock-runtime` API. On AWS that is the real service. Locally, set
`BEDROCK_LOCAL=1` and the same calls run against Ollama through a small
Bedrock-shaped shim the book owns, so you can develop offline and switch to
Bedrock by unsetting one variable.

## Run it

```
make up          # start the vector store
make seed        # generate the knowledge base and embed it into pgvector
make guardrail   # create the Bedrock guardrail (once)
make ask         # a grounded, cited answer to one question, guardrail on
make decide      # the agent works a whole deal and routes it
make serve       # the agent on the AgentCore contract (POST /invocations :8080)
make down        # stop and clean
```

Work your own deal:

```
make decide DEAL="VoltStack, a battery storage integrator, seeks a 12 million US dollar facility."
```

## Local vs AWS

The code never changes between local and AWS. The vector store moves with the
`PG*` variables: the local container, or the Aurora endpoint (a short-lived IAM token on AWS). The models move with
`BEDROCK_LOCAL`: unset it for Bedrock, set it for Ollama.

To run fully offline, install Ollama and pull the two models:

```
ollama pull qwen3:0.6b
ollama pull mxbai-embed-large
BEDROCK_LOCAL=1 make seed ask decide
```

`mxbai-embed-large` returns 1024-dimensional vectors, the same width as Titan
Text Embeddings v2, so the vector store schema is identical either way. The
`aws/` folder deploys the assistant on real AWS: Aurora PostgreSQL Serverless (express configuration) and a Lambda.

## Files

The shared seam and each chapter element live under `src/`; the `make` targets
set `PYTHONPATH` to `src/` so the elements import the seam as top-level modules
(a direct run is `PYTHONPATH=src uv run src/knowledge-base/corpus.py`).

- `src/models.py`: the model seam, one Bedrock-shaped interface over Bedrock or Ollama
- `src/stores.py`: the vector store seam, retrieval over pgvector
- `src/knowledge-base/corpus.py`: generate the knowledge base and embed it
- `src/underwriting-agent/authority.py`: the delegated-authority routing logic
- `src/underwriting-agent/guardrails.py`: the Bedrock guardrail and the local shim
- `src/underwriting-agent/agent.py`: the ask, decide, and serve entry points
- `local/rag-stack.yml`: the local vector store (pgvector)
- `aws/`: deploy on real AWS
- `Makefile`: the targets above; `make lint` runs ruff
