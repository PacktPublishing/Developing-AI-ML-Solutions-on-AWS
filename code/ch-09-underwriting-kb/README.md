# The underwriting knowledge base

The local retrieval path runs against OpenSearch, with grounded answers and
citations. The underwriter interface is a FastAPI ask/recommend app (`app/`) that
runs under uvicorn locally and, unchanged, in a container Lambda behind API
Gateway on AWS. The cloud round provisions the OpenSearch Service domain (with
Dashboards) and the app.

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
make up                       # the local OpenSearch node and Dashboards
make gen                      # write the synthetic memo corpus (reproducible from --seed)
BEDROCK_LOCAL=1 make seed      # embed the memos into the store via Ollama
BEDROCK_LOCAL=1 make ask  Q="How is DTI assessed for a grocery business?"
BEDROCK_LOCAL=1 make cases DEAL="A logistics firm seeks a 500,000 dollar working-capital loan"
make statements               # the synthetic bank statement PDFs to review
BEDROCK_LOCAL=1 make review CASE=CASE-20260000
make down                     # tear it down
```

`make ask` retrieves the nearest memo chunks and answers the question grounded in
them, citing each claim's source loan; `make cases` gathers the most similar
prior cases into a draft recommendation the underwriter signs off on. Unset
`BEDROCK_LOCAL` to run the embeddings and generation on Amazon Bedrock instead.
`make review` is the underwriter's own loop: it uploads an applicant's bank
statement, totals it, indexes it in OpenSearch beside the archive, and asks the
agent what the numbers allow and whether the bank has lent on similar terms
before. The statement is stored with `source: submission` and the archive search
filters on `source: corpus`, so a document under review is never returned as its
own precedent, and `--keep` aside it is deleted once the review is done.

Totalling the statement is arithmetic, so it is a parser and not a prompt: the
agent reads the income and existing repayments back out of the index rather than
adding up a table it retrieved. That is the same rule the affordability tool
follows, and for the same reason.

Expect terse answers on a small local model, and note that tool calling needs
Qwen3 4b or larger: at 0.6b the agent replies that the statement is unavailable
instead of calling the tool. Bedrock is the graded path.

**OpenSearch Dashboards** comes up at http://localhost:5601, the same search UI
Amazon OpenSearch Service hosts on the domain, so the browser experience is
identical locally and on AWS. Open Discover on the `memo_chunks` index, or run a
k-NN query in Dev Tools.

## Layout

- `etl/gen_memos.py`: the synthetic-memo generator, two shapes, `--messy` for
  the real-world typos and blank fields an extractor must tolerate
- `etl/embed_memos.py`: chunk each memo and embed it into the index
- `etl/gen_statements.py`: synthetic bank statement PDFs, with a manifest of the
  figures each one encodes so the parser can be checked against them
- `src/models.py`: the model seam — one Bedrock-shaped interface, Ollama behind it
- `src/stores.py`: the store seam — one function naming the backend
- `src/opensearch_store.py`: the OpenSearch backend — local container, Amazon
  OpenSearch Service on AWS
- `src/retrieve.py`: grounded question answering and case assembly, with citations
- `src/affordability.py`: the DTI arithmetic the agent calls as a tool
- `src/statements.py`: read an uploaded statement PDF, total it, and index it
- `src/review.py`: upload one statement and print the agent's review of it
- `src/agent.py`: the Strands agent, memo search plus the statement and the
  affordability tools
- `local/kb-stack.yml`: the compose stack — the OpenSearch node and Dashboards
- `tests/test_opensearch.py`: the score conversion, and a round-trip when a node
  is up
- `tests/test_affordability.py`: the instalment, the ratio, and its inverse
- `tests/test_statements.py`: the parser against the generator's own manifest

## AWS services and local stand-ins

- **AWS services:** Amazon Bedrock (embeddings, Converse), Amazon OpenSearch
  Service, AWS Lambda, Amazon API Gateway
- **Local stand-ins:** Ollama (Bedrock), OpenSearch in a container

## The underwriter app

`app/main.py` is one FastAPI service: it serves the ask/recommend page with
`app.frontend()` and exposes `/ask` and `/cases`, reusing the retrieval seam.
`make app` runs it under uvicorn on http://localhost:8080 over the selected
store; the same image (with the AWS Lambda Web Adapter, see `Dockerfile`) runs in
a container Lambda behind API Gateway on AWS. See `aws/` for the cloud round.
