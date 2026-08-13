# Agents for customer conversation classification

The whole flow is verified both locally (from-source shims) and on AWS (real
Bedrock batch jobs + real SNS): a batch job classified all 320 conversations with
Claude Haiku 4.5, routed them to ten real SNS topics, and scored 0.69 accuracy.

The pile every support and collections team knows: thousands of conversations
nobody has time to skim. Each one is classified into a category with zero-shot
inference and handed to the right team through SNS, with a labeled set keeping the
accuracy honest.

The work runs as an **Amazon Bedrock batch inference** job: every conversation
becomes one line of a JSONL manifest in S3, Bedrock replays each `modelInput`
against the model and writes the answers back as `.jsonl.out`, and we collect and
route them. Batch is the right tool for volume — it is asynchronous and roughly
half the on-demand price — where interactive concurrency is not needed.

## Run it locally

```
make up                       # S3Proxy (the batch manifests + outputs live here)
make prepare HF_TOKEN=...     # build the labeled datasets (RetailBanking is gated)
make classify                 # run the batch job (local shim) and score it
make route                    # publish each conversation to its team's topic
make eval                     # accuracy, macro-F1, multiclass Brier, skill score
make down
```

Locally, **Bedrock batch is a from-source shim**: `create_model_invocation_job`
runs as a background thread that reads the JSONL shards from S3Proxy, invokes the
model on each record through the Ollama-backed model seam, and writes `.jsonl.out`
— the same async submit/poll/collect flow as the real service (and the same design
LocalStack's own Bedrock emulator uses: it too runs the job over Ollama). **SNS is
a from-source shim** that fans each publish out to its subscribers, delivering to a
file inbox per team. On a small local model (Qwen3 0.6b) the accuracy is
illustrative; the graded numbers come from Claude on Bedrock in the cloud round.

## Datasets

- [`oopere/RetailBanking-Conversations`](https://huggingface.co/datasets/oopere/RetailBanking-Conversations)
  — 320 multi-turn conversations over ten topics (the routing categories).
- [`legacy-datasets/banking77`](https://huggingface.co/datasets/legacy-datasets/banking77)
  — 3,080 short queries over 77 intents (the harder eval).

## Layout

- `etl/prepare.py`: group RetailBanking turns into labeled transcripts; load banking77
- `src/models.py`: the model seam — one Converse interface, Bedrock or Ollama
- `src/batch.py`: the batch-inference seam — real Bedrock, or the local from-source
  batch shim over S3Proxy + Ollama
- `src/classify.py`: build the Converse manifest, submit the job, collect, map to categories
- `src/sns.py` + `src/route.py`: the routing seam — real SNS, or the local file-inbox shim
- `src/evaluate.py`: accuracy, macro-F1, and the raw multiclass Brier plus a skill score

## The Brier Index, one-vs-rest

The **Brier Index** (Forecasting Research Institute, 2026), `100 * (1 - sqrt(B))`,
rescales a Brier score so higher is better (100 = perfect, 50 = always forecasting
the base event, 0 = maximally wrong). Its anchors assume a 50% base rate, and the
joint multiclass Brier ranges over `[0, 2]` with anchors that drift with `K` -- so
we apply the index **one-vs-rest**, one binary problem per class, which keeps each
Brier in `[0, 1]` with the fixed anchors and localizes which categories are
miscalibrated.

To stay fair under class imbalance we also report the **Adjusted Brier Index**,
referenced to each class's own base rate `p`:

    Adjusted Brier Index = 100 - 50 * sqrt(Brier / (p * (1 - p)))

A rare class with a low raw Brier is not automatically good; it is judged against
`p * (1 - p)`. The swing is largest under imbalance (a credit or fraud rate), where
raw Brier flatters the rare class. These indices are only meaningful with soft
probabilities: run `classify --probs` so the model emits a distribution; with hard
labels the Brier is just twice the error rate.

## AWS services and local stand-ins

- **AWS services:** Amazon Bedrock (batch inference, Converse), Amazon SNS, Amazon S3
- **Local stand-ins:** a from-source Bedrock batch shim over S3Proxy + Ollama, a
  from-source SNS shim (file inboxes), S3Proxy (S3)

## Run it on AWS

```
make -C aws iam            # ch10-user (privileged identity), then assume it via a profile
make -C aws deploy         # the batch bucket + the bedrock.amazonaws.com service role
make -C aws classify       # a real CreateModelInvocationJob (tens of minutes)
make -C aws route          # publish to real SNS topics
make -C aws eval
make -C aws teardown
```

The submitting user needs `CreateModelInvocationJob` + `InvokeModel`, and the batch
job invokes the model **as the service role**, so that role needs `InvokeModel` too.
Manifest sharding (1 GiB/file, 5 GiB/job) is the remaining piece for very large jobs.
