# Agents for customer conversation classification

> **In progress.** The local path — batch inference over the two datasets, routing
> through SNS, and the eval — is built and verified. The AWS round (real Bedrock
> batch jobs + real SNS) is still to come.

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

## A note on the Brier score

We report the **raw multiclass Brier score**, not the binary "Brier Index". Its
100/50/0 anchors are calibrated to the two-class Brier score; the multiclass score
ranges over `[0, 2]` and an uninformed forecast over `K` classes gives `1 - 1/K`,
so the anchors drift with `K`. With hard labels the multiclass Brier is exactly
twice the error rate, so the **skill score** against a uniform baseline is the more
honest summary; soft probability outputs would make the Brier non-degenerate.

## AWS services and local stand-ins

- **AWS services:** Amazon Bedrock (batch inference, Converse), Amazon SNS, Amazon S3
- **Local stand-ins:** a from-source Bedrock batch shim over S3Proxy + Ollama, a
  from-source SNS shim (file inboxes), S3Proxy (S3)

## Still to come

- the AWS round: real `CreateModelInvocationJob` (with the `bedrock.amazonaws.com`
  batch service role) + real SNS topics, and manifest sharding for large jobs
