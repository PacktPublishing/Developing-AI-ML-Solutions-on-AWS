# The underwriting knowledge base

> **Scaffold in progress.** This chapter is being built on `ch09-ch12`; the
> layout below is the plan, not yet a verified run. Sections fill in as each
> piece is written and tested locally, then on AWS.

The knowledge base the underwriting agent stands on: policy documents, credit
reports, and past dossier files, embedded with Bedrock and stored for retrieval,
so a question is answered with citations to the source and an agent can assemble
similar past cases into a recommendation the underwriter signs off on.

## What it builds

- A corpus of **synthetic underwriting memos** — generated from scratch, no
  customer names or PII, amounts in `$`, no real company names — standing in for
  the private dossier files the real system reads.
- **Bedrock embeddings** over that corpus, stored two ways for comparison:
  **Aurora PostgreSQL + pgvector** and **Amazon S3 Vectors**.
- Retrieval with **citations to the source memo**, and an agent that gathers the
  most similar prior cases into a draft recommendation.
- A retrieval interface — the shipped option is still open (a small Lambda
  serving HTML through API Gateway to Bedrock, versus an OpenSearch-backed search
  UI); **OpenSearch** is exercised both locally and on AWS.

## Planned layout

- `etl/`: the synthetic-memo generator and the embed-and-load step
- `local/`: compose stack — pgvector Postgres (Aurora), OpenSearch, Ollama as the
  local Bedrock stand-in
- `src/`: the retrieval + recommendation code, tested here first
- `aws/template.yaml`: SAM — Aurora Serverless v2 + pgvector, S3 Vectors,
  OpenSearch, the retrieval Lambda + API Gateway
- `diagrams/`: cloud architecture and its local mirror

## AWS services and local stand-ins

- **AWS services:** Amazon Bedrock (embeddings, Converse), Amazon Aurora
  PostgreSQL + pgvector, Amazon S3 Vectors, Amazon OpenSearch Service, AWS
  Lambda, Amazon API Gateway
- **Local stand-ins:** Ollama (Bedrock), pgvector Postgres (Aurora), OpenSearch
  in a container
