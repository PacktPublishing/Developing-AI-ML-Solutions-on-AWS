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

The same conversations route through a decision model instead, which needs no
batch job and no S3 at all:

```
make decide VARIANT=b         # one typed question per conversation
make route                    # the same routing, from the same predictions file
make eval                     # the same scoring, on a real distribution
make decide VARIANT=b THRESHOLD=0.6   # hold the close calls back for a person
```

### What the container measured

The agent is the same file locally and deployed, but the packaging is not, and
the packaging is where AgentCore's limits bite. Building the image on a laptop
answers that before any AWS spend (measured on arm64; the deployed image is
x86_64, because every GPU family AgentCore supports is):

| | result | AgentCore limit |
| --- | --- | --- |
| image, default aarch64 torch (bundles CUDA) | 9.39 GB | 2 GB, not adjustable |
| image, CPU-only torch | **1.47 GB** | fits |
| model weights | 4.8 GB | too big for either artifact |
| CPU inference, peak resident memory | **7.37 GB** | microVM session caps at 8 GB |

Three things follow, and none of them needed an AWS account:

- **The CUDA image cannot be deployed at all.** 3.3 GB of `nvidia` and 916 MB of
  `triton` come from the default aarch64 torch wheel. The CPU index drops both.
- **The weights cannot ship inside the artifact.** A container image stops at
  2 GB and an S3 source package at 750 MB unpacked, against 4.8 GB of weights,
  so they are fetched to a mounted volume at session start. That is the one-time
  cost the chapter's session handling is built around.
- **microVMs are out.** The CPU image fits the image limit, but inference peaks
  at 7.37 GB against a hard 8 GB session cap, which leaves nothing for the
  runtime, and a microVM gives 2 vCPU. Running the CPU image under those limits
  locally was killed with `OOMKilled=true`. So the chapter deploys on the
  Instances compute type with a GPU capacity provider.

The engine upcasts the torso to fp32 on CPU, which is where the 7.37 GB comes
from: a 2B model is 4.5 GB in bf16 and roughly double that as fp32.

### Swapping the model

A decision model is an interface, not a product, so the chapter treats it as one.
`DECIDER_BACKEND` picks where the answer comes from:

| backend | what answers | needs |
| --- | --- | --- |
| `local` | the model in this process | nothing but the checkpoint |
| `http` | any System One server | `DECIDER_URL` |
| `agentcore` | an AgentCore Runtime endpoint | `AGENT_RUNTIME_ARN` |


The `http` backend is the interesting one. Strands Decider and
[Kev](https://github.com/jaredpalmer/kev) both serve `POST /v1/systemone` with
the same request and the same response, so changing which model routes your
conversations is changing a URL:

```
make serve-decider                                     # the baseline, port 8019
make decide DECIDER_BACKEND=http DECIDER_URL=http://127.0.0.1:8019

# or point at Kev instead, started from its own repo, and change nothing else
#   uv run -m kev.serve --run jaredpalmer/kev-4b --port 8019
make decide DECIDER_BACKEND=http DECIDER_URL=http://127.0.0.1:8019
```

Every prediction records the `model` that answered and the `backend` it came
from, taken from the response rather than from configuration, so two runs remain
comparable after the fact.

TypeSafe's Jev is the one that cannot be swapped in this way. It is a hosted API
with no released weights, so there is nothing to run behind `http` or to put in
an AgentCore container, and the conversations would leave your account to reach
it. The chapter uses it as a contrast, not as a backend.

Locally, **Bedrock batch is a from-source shim**: `create_model_invocation_job`
runs as a background thread that reads the JSONL shards from S3Proxy, invokes the
model on each record through the Ollama-backed model seam, and writes `.jsonl.out`
— the same async submit/poll/collect flow as the real service (and the same design
LocalStack's own Bedrock emulator uses: it too runs the job over Ollama). **SNS is
a from-source shim** that fans each publish out to its subscribers, delivering to a
file inbox per team. On a small local model (Qwen3 0.6b) the accuracy is
illustrative; the graded numbers come from Claude on Bedrock in the cloud round.

### AgentCore runs locally too

`agentcore/app.py` is the deployable unit, and it is also the local one.
`BedrockAgentCoreApp` serves `POST /invocations` and `GET /ping`, which is the
contract AgentCore Runtime invokes a container on, so the same file answers on a
laptop and in the cloud. The local round is the deployed path, not a stand-in
for it:

```
make serve-agentcore                                   # http://127.0.0.1:8080
make decide DECIDER_BACKEND=http \
  DECIDER_URL=http://127.0.0.1:8080 DECIDER_PATH=/invocations
```

The payload is a System One request, so the `http` backend reaches the AgentCore
app with no client change, and `DECIDER_PATH=/invocations` runs the round through
the exact path the cloud uses. The AgentCore CLI has its own loop over the same
file, `agentcore dev` for a reloading server with `agentcore invoke --local`,
then `agentcore configure -e agentcore/app.py` and `agentcore launch`.

The weights load on the first invocation of a container, tens of seconds, and
are reused after that. That is why the router holds one session id for a whole
run: the first call pays for the model and the rest do not.

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

## The decision model

[Strands Decider 2B](https://strandsagents.com/blog/introducing-strands-decider/)
keeps a Qwen3.5-2B-Base torso, drops the text-generation head, and puts a pointer
head of about a million parameters in its place. That head scores the options a
question supplies in one forward pass, and a softmax over those scores is the
answer, so a `choice` question comes back as the chosen option, a probability for
every option, and a confidence. There is no text, which is why there is nothing to
parse and nothing to truncate: the failure modes the batch path has to defend
against cannot arise here.

Confidence is normalised max-probability, `(N * p_max - 1) / (N - 1)`, so it means
the same thing for three options as for ten and one `THRESHOLD` keeps holding as
the team list grows.

Two things worth knowing before building on it:

- The checkpoint is Apache-2.0 and is two downloads, not one: the adapter and head
  are about 0.235 GB, and the Qwen3.5-2B-Base torso underneath is another 4.571 GB,
  fetched on first load. That split is why the AWS side keeps weights on a volume
  rather than in a container image.
- `strands-decider` is version 0.3.0 and its own description calls it an
  experimental typed classifier. Pin it, and expect the API to move.

Latency scales with the length of the state, so the sub-200 ms figures quoted for
short tasks are not what a multi-turn transcript costs; see the numbers below.

## Two routers, one question

Both routers answer the same question about the same 320 conversations, briefed
from the same definitions in `src/categories.py`. Measured on 2026-11-07, the
decision model on an M3 laptop and the zero-shot LLM through the local batch shim:

| router | accuracy | macro-F1 | Brier | Brier Index | coverage |
| --- | --- | --- | --- | --- | --- |
| decision model, variant b | 0.666 | 0.666 | 0.515 | 78.6% | 1.000 |
| zero-shot LLM, variant a (Qwen3 0.6b) | 0.156 | 0.059 | 1.687 | 61.9% | 0.997 |

The accuracy gap is not the interesting part, because the local LLM is a 0.6b
model and Claude Haiku 4.5 scored 0.69 on these same conversations in the cloud
round above. The interesting part is that a 2B model reading nothing but the
conversation, on a laptop, with no job to submit and no answer to parse, lands
within three points of it.

Two differences matter more than the accuracy:

- **Where the predictions go.** The LLM collapsed 223 of 320 conversations into
  Bank Accounts and 96 into Investment Funds, leaving seven of the ten teams with
  nothing at all, which is why its macro-F1 is 0.059 against an accuracy of 0.156.
  The decision model put conversations in every queue.
- **What the probabilities are worth.** The LLM has to be talked into emitting a
  distribution and then checked, and on the local model it mostly fails: asked for
  one, Qwen3 0.6b answered `1.0` for several categories at once and 50 of 60
  records were refused. The decision model's distribution is a softmax over the
  options, so it always sums to one and there is nothing to validate.

Describing the team borders is worth about five points (62.5% to 67.5% over the
first 40 conversations), the same direction the prompt variants move the LLM.

Confidence is the part with no LLM equivalent. Holding back the conversations the
model is least sure of trades coverage for quality, and the chapter's own metrics
show the trade: at `THRESHOLD=0.6`, coverage falls to 0.800 and the Brier over
what remains improves from 0.496 to 0.409. Those eight conversations in a hundred
are the ones worth a person's time.

Latency is about 1.9 s per conversation on an M3 across the full 320, around ten
minutes for the set. That is far off the sub-200 ms figures quoted for short
tasks, because a multi-turn transcript is a long state and latency grows with it.

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
- `agentcore/app.py`: the decision model behind the AgentCore Runtime contract,
  the same file locally and deployed
- `src/categories.py`: where each team's work begins and ends, written once and
  read by both routers, so comparing them compares the routers
- `src/decider.py`: the decision-model seam, either the model loaded in this
  process or an AgentCore Runtime endpoint
- `src/decide.py`: ask one typed question per conversation, route on the answer,
  and hold back the close calls
- `src/runs.py`: one identifier per run, so two runs' outputs can be told apart
- `src/sns.py` + `src/route.py`: the routing seam — real SNS, or the local file-inbox shim
- `src/evaluate.py`: accuracy and coverage, macro-F1, and the raw multiclass
  Brier plus a skill score, over the records that produced a forecast
- `tests/test_classify.py`: the output file, the answers that are refused, and
  the job statuses that mean a job has stopped moving
- `tests/test_evaluate.py`: the scoring maths, and what a missing prediction
  does and does not score
- `tests/test_decide.py`: the question put to the decision model, and the
  confidence policy applied to its answer

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

- **AWS services:** Amazon Bedrock (batch inference, Converse), Amazon Bedrock
  AgentCore Runtime (the decision model, on the Instances compute type), Amazon
  SNS, Amazon S3
- **Local stand-ins:** a from-source Bedrock batch shim over S3Proxy + Ollama, a
  from-source SNS shim (file inboxes), S3Proxy (S3). The decision model has no
  stand-in at all: `agentcore/app.py` serves the AgentCore Runtime contract
  itself, so the local round runs the file that deploys. It also needs no
  containers, where the batch path needs S3Proxy for its manifests.

## Run it on AWS

```
make -C aws iam            # ch10-user (privileged identity), then assume it via a profile
make -C aws deploy         # the batch bucket + the bedrock.amazonaws.com service role
make -C aws classify       # a real CreateModelInvocationJob (tens of minutes)
make -C aws route          # publish to real SNS topics
make -C aws eval
make -C aws teardown
```

### The decision model on AgentCore

Off by default, because it provisions a VPC and GPU EC2 in your account that
bill by the hour while a session is alive.

```
make -C aws decider-round      # the four steps below, in order
```

It is four steps because the runtime needs an image that the same stack's
repository holds:

```
make -C aws deploy WITH_DECIDER=true     # VPC, repository, CodeBuild, roles, capacity
make -C aws image                        # CodeBuild builds x86_64 and pushes
make -C aws deploy WITH_DECIDER=true DECIDER_IMAGE=<repo>:latest   # the runtime
make -C aws prepare                      # torch and the weights onto the volume
make -C aws decide                       # route the dataset through it
```

`prepare` is the step with no local equivalent, and it exists because of the
limits in the table above: the image carries the agent and the runtime SDK only,
so the first call of a session installs CUDA torch and fetches the checkpoint
onto the mounted EBS volume. It takes minutes. The volume survives session
stops, so a resumed session skips it, which is also why `src/decider.py` holds
one session id for a whole run rather than one per conversation.

Two names in the template cannot take hyphens, and the validation error does not
say so: the capacity provider and the runtime are `ch10_decider_capacity` and
`ch10_decider`.

Expect `InsufficientInstanceCapacity` on GPU instances. The capacity provider
allows `g6.xlarge` and `g5.xlarge` across two availability zones for that reason;
add more types with `DeciderInstanceTypes` if both are short.

The submitting user needs `CreateModelInvocationJob` + `InvokeModel`, and the batch
job invokes the model **as the service role**, so that role needs `InvokeModel` too.
Manifest sharding (1 GiB/file, 5 GiB/job) is the remaining piece for very large jobs.
