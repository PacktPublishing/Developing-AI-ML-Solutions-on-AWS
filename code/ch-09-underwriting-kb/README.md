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
make down                     # tear it down
```

`make ask` retrieves the nearest memo chunks and answers the question grounded in
them, citing each claim's source loan; `make cases` gathers the most similar
prior cases into a draft recommendation the underwriter signs off on. Unset
`BEDROCK_LOCAL` to run the embeddings and generation on Amazon Bedrock instead.
On a small local model (Qwen3 0.6b) expect terse answers; Bedrock is the graded
path.

**OpenSearch Dashboards** comes up at http://localhost:5601, the same search UI
Amazon OpenSearch Service hosts on the domain, so the browser experience is
identical locally and on AWS. Open Discover on the `memo_chunks` index, or run a
k-NN query in Dev Tools.

## Layout

- `etl/gen_memos.py`: the synthetic-memo generator, two shapes, `--messy` for
  the real-world typos and blank fields an extractor must tolerate
- `etl/embed_memos.py`: chunk each memo and embed it into the index
- `src/models.py`: the model seam — one Bedrock-shaped interface, Ollama behind it
- `src/stores.py`: the store seam — one function naming the backend
- `src/opensearch_store.py`: the OpenSearch backend — local container, Amazon
  OpenSearch Service on AWS
- `src/retrieve.py`: grounded question answering and case assembly, with citations
- `src/affordability.py`: the DTI arithmetic the agent calls as a tool
- `src/agent.py`: the Strands agent — memo search plus the affordability tools
- `local/kb-stack.yml`: the compose stack — the OpenSearch node and Dashboards
- `tests/test_opensearch.py`: the score conversion, and a round-trip when a node
  is up
- `tests/test_affordability.py`: the instalment, the ratio, and its inverse

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
