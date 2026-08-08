# Agents for customer conversation classification

> **Scaffold in progress.** This chapter is being built on `ch09-ch12`; the
> layout below is the plan, not yet a verified run. Sections fill in as each
> piece is written and tested locally, then on AWS.

The pile every support and collections team knows: thousands of conversations
nobody has time to skim. A Strands and Bedrock agent files each one into a
category with zero-shot classification and hands it to the right team through
SNS, with a labeled set keeping the accuracy honest.

## What it builds

- A **zero-shot classifier** agent — Strands over Bedrock — reusing the local
  model seam from chapter 3 (Ollama offline, Bedrock on AWS).
- **Batch inference** over a JSONL manifest: the offline path that scores
  thousands of conversations at once rather than one Converse call each.
- **Routing through SNS** — each classified conversation published to the topic
  its category subscribes to; locally an SNS shim composed from source (modelled
  on the reference in `localstack-pro`), on AWS the real service.
- A **labeled evaluation set** to keep accuracy honest.

### Datasets

- [`oopere/RetailBanking-Conversations`](https://huggingface.co/datasets/oopere/RetailBanking-Conversations)
- [`legacy-datasets/banking77`](https://huggingface.co/datasets/legacy-datasets/banking77)

### A note on evaluation

Report the **raw multiclass Brier score** (or log loss) and, if useful, a skill
score against a majority-class or uniform baseline — not the binary-anchored
"Brier Index". Its 100/50/0 anchors are calibrated to the *binary* Brier score;
the multiclass score ranges over `[0, 2]` and an uninformed forecast over `K`
classes gives `1 - 1/K`, so the 50 anchor drifts with the number of classes and
stops being comparable across taxonomies. With hard labels (not probability
vectors) it degenerates to twice the error rate — a monotone rescale of accuracy
that adds no information.

## Planned layout

- `etl/`: fetch the datasets, build the batch JSONL manifest and the labeled set
- `local/`: compose stack — Ollama (Bedrock), the from-source SNS shim
- `src/`: the classifier agent, the batch runner, the routing + evaluation code
- `aws/template.yaml`: SAM — Bedrock batch inference, SNS topics + subscriptions
- `diagrams/`: cloud architecture and its local mirror

## AWS services and local stand-ins

- **AWS services:** Amazon Bedrock (batch inference, Converse), Amazon SNS,
  Amazon S3
- **Local stand-ins:** Ollama (Bedrock), an SNS shim composed from source, a
  local batch-inference runner over the JSONL manifest
