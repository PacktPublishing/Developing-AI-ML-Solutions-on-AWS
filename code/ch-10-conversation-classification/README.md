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

## What a batch job does to a result

A batch job answers hours later, in a file, and most of what can go wrong goes
wrong per record rather than per job. Three things follow from that.

**One failed record is one failed record.** Bedrock writes an output line per
conversation, and a line for a record it could not run carries an error in place
of `modelOutput`. Collection records that against the record and carries on, so
one throttled conversation does not cost the other three hundred and nineteen.
The reasons are kept, written into the predictions as a `failed` field, and
counted in the run's own summary.

**A job that did not finish does not get to replace the predictions on disk.**
The statuses that mean a job has stopped moving are Completed,
PartiallyCompleted, Failed, Stopped and Expired, and only the first two leave
output worth reading. A Failed job used to be collected anyway, which produced an
empty prediction for every conversation and wrote them over the last good run.
PartiallyCompleted is read, and the report says that not every record was
processed.

**Every run is its own run.** The manifest key, the job name and the output prefix
all carry a run id, so classifying a second dataset or a second prompt variant
while the first job is still queued no longer replaces the file that job is about
to read. Pass `--out` to keep a variant's predictions beside the canonical ones.

An answer is read rather than guessed at. A category has to be named exactly,
after the formatting a model wraps a one word answer in comes off, and an answer
that mentions several categories is a failed record rather than a vote for
whichever is listed first: "Not Cards. Digital Banking." is not Cards. A
probability has to be a finite number in [0,1], with the categories totalling
about one, so "not sure", null, 1e309 and a JSON object that did not parse are
failed records instead of a crash or a vector of zeroes.

`make eval` reports coverage beside accuracy for the same reason. Accuracy counts
every conversation, missing ones included, because a router that answered nothing
routed nothing. The Brier score and both indices are computed over the records
that produced a forecast, because a record with no prediction has no distribution
to score, and scoring it as an all-zero vector used to flatter it above a wrong
answer that at least committed to something.

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
- `src/evaluate.py`: accuracy and coverage, macro-F1, and the raw multiclass
  Brier plus a skill score, over the records that produced a forecast
- `tests/test_classify.py`: the output file, the answers that are refused, and
  the job statuses that mean a job has stopped moving
- `tests/test_evaluate.py`: the scoring maths, and what a missing prediction
  does and does not score

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
