# The underwriting knowledge base

> **In progress.** The local retrieval path is built and verified against all
> three stores — pgvector, OpenSearch, and S3 Vectors — with grounded answers and
> citations. The AWS cloud round (Aurora, Amazon OpenSearch Service, S3 Vectors)
> and the retrieval interface are still to come.

The knowledge base the underwriting agent stands on: policy documents, credit
reports, and past dossier memos, embedded with Bedrock and stored for retrieval,
so a question is answered with citations to the source and an agent can assemble
similar past cases into a recommendation the underwriter signs off on.

The memos are **synthetic** — generated from scratch, no real names, PII, or
company data; amounts in USD. They reproduce the two shapes a real corpus has: a
structured 8-section credit memo with an extractable financial-snapshot table,
and a nested email approval thread that is mostly boilerplate around a few lines
of reasoning.

## Run it locally

```
make up                       # pgvector, the local vector store
make gen                      # write the synthetic memo corpus (reproducible from --seed)
BEDROCK_LOCAL=1 make seed      # embed the memos into the store via Ollama
BEDROCK_LOCAL=1 make ask  Q="How is DTI assessed for a grocery business?"
BEDROCK_LOCAL=1 make cases DEAL="A logistics firm seeks a 500,000 dollar working-capital loan"
make down                     # tear it down
```

`make ask` retrieves the nearest memo chunks and answers the question grounded in
them, citing each claim's source loan; `make cases` gathers the most similar
prior cases into a draft recommendation the underwriter signs off on. Unset
`BEDROCK_LOCAL` to run the embeddings and generation on Amazon Bedrock instead.
On a small local model (Qwen3 0.6b) expect terse answers; Bedrock is the graded
path.

The same commands run against **OpenSearch** or **S3 Vectors** instead of
pgvector by setting `STORE`, which starts that store's container (a compose
profile) — the retrieval code is identical for all three:

```
STORE=opensearch make up   && STORE=opensearch BEDROCK_LOCAL=1 make seed
STORE=s3vectors  make up   && STORE=s3vectors  BEDROCK_LOCAL=1 make seed
STORE=s3vectors  BEDROCK_LOCAL=1 make ask Q="..."
```

Locally, S3 Vectors is a from-source shim over S3Proxy: each vector is one S3
object and the query is a brute-force cosine scan, mirroring the real service's
create_vector_bucket / create_index / put_vectors / query_vectors. Unset
`S3VECTORS_LOCAL` (and `BEDROCK_LOCAL`) to talk to the real `s3vectors` API.

## Layout

- `etl/gen_memos.py`: the synthetic-memo generator, two shapes, `--messy` for
  the real-world typos and blank fields an extractor must tolerate
- `etl/embed_memos.py`: chunk each memo and embed it into the selected store
- `src/models.py`: the model seam — one Bedrock-shaped interface, Ollama behind it
- `src/stores.py`: the store seam — one interface, `STORE` picks the backend
- `src/pgvector_store.py`: the pgvector backend — local Postgres, Aurora on AWS
- `src/opensearch_store.py`: the OpenSearch backend — local container, Amazon
  OpenSearch Service on AWS
- `src/s3vectors_store.py`: the S3 Vectors backend, plus its local shim over
  S3Proxy — Amazon S3 Vectors on AWS
- `src/retrieve.py`: grounded question answering and case assembly, with citations
- `local/kb-stack.yml`: the compose stack — pgvector, with OpenSearch and S3Proxy
  each behind a profile
- `tests/test_pgvector.py`: a pgvector round-trip that provisions its own
  container and stubs the embeddings, so it needs no cloud

## AWS services and local stand-ins

- **AWS services:** Amazon Bedrock (embeddings, Converse), Amazon Aurora
  PostgreSQL + pgvector, Amazon S3 Vectors, Amazon OpenSearch Service, AWS
  Lambda, Amazon API Gateway
- **Local stand-ins:** Ollama (Bedrock), pgvector Postgres (Aurora), OpenSearch
  in a container, a from-source shim over S3Proxy (S3 Vectors)

## Still to come

- the retrieval **interface** (open: a small Lambda serving HTML through API
  Gateway to Bedrock, versus an OpenSearch-backed search UI)
- `aws/template.yaml` and the cloud round (Aurora, OpenSearch Service, S3 Vectors)
