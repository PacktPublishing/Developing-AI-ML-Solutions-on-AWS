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
the packaging is where the limits bite. Building the image on a laptop answers
that before any AWS spend (measured on arm64; the deployed model image is
x86_64, because every GPU family is):

| | result | the limit it meets |
| --- | --- | --- |
| image, default torch (bundles CUDA) | 9.39 GB | AgentCore caps an image at 2 GB |
| image, CPU-only torch | 1.47 GB | fits, but see the next row |
| CPU inference, peak resident memory | 7.37 GB | an AgentCore microVM session caps at 8 GB |
| model weights | 4.8 GB | too big for a 2 GB image or a 750 MB code zip |

That is why this chapter runs on two services rather than one. **The model goes
to SageMaker**, which has no image limit and real GPU instances. **The agent goes
to AgentCore Runtime**, which is what it is for, and which a few kilobytes of
routing policy fits comfortably on a serverless microVM.

Running the CPU image under a microVM's limits locally ends in
`OOMKilled=true`, which is the measurement rather than the guess. The engine
upcasts the torso to fp32 on CPU, so a 2B model that is 4.5 GB in bf16 is
roughly double that.

Serverless Inference on SageMaker is not an option either, for the same
arithmetic: its ceiling is 6144 MB and it offers no GPU.

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

### Swapping the checkpoint

`DECIDER_MODEL` names the checkpoint the `local` and `http` backends load, and
`model/Dockerfile` takes the same name as `CHECKPOINT` for the endpoint image.
Two sizes are published as System One deciders. Both answer this chapter's
question with one output token, and the difference between them is latency and
download size rather than accuracy:

| checkpoint | accuracy, variant a / b | median latency, a / b | base model |
| --- | --- | --- | --- |
| `strands-decider-2B-qwen3.5-v1-2610` | 0.684 / 0.666 | 1.2 s / 1.3 s | `Qwen/Qwen3.5-2B-Base`, 4.5 GB |
| `strands-decider-E4B-gemma4-v1-2610` | 0.669 / 0.691 | 4.4 s / 5.2 s | `google/gemma-4-E4B-it`, 16 GB |

All 320 conversations, `local:mps`, both checkpoints loaded by the same engine:

```
DECIDER_MODEL=StrandsAgents/strands-decider-E4B-gemma4-v1-2610 make decide VARIANT=b
```

McNemar on the paired predictions does not separate the checkpoints: 3 against 11
at variant b (p=0.061) and 11 against 6 at variant a (p=0.332). The chapter stays
on the 2B because four times the latency and a 16 GB base buy no accuracy that
320 conversations can measure. Run the command above if you want to see that for
yourself rather than take it from this table.

Each published decider is a LoRA adapter, and it names the base it was trained
over in its own `strands_decider_config.json`. The image build reads that field,
so a checkpoint is the only thing you change.

### AgentCore runs locally too

`agent/main.py` is the deployable unit and the local one. `BedrockAgentCoreApp`
serves `POST /invocations` and `GET /ping`, which is the contract AgentCore
Runtime invokes a container on, and also exactly what SageMaker requires of an
inference container. One contract, two services, no stand-in:

```
ENDPOINT_NAME=ch10-decider uv run agent/main.py     # http://127.0.0.1:8080
curl -X POST http://127.0.0.1:8080/invocations -H 'content-type: application/json' \
  -d '{"text": "my debit card was declined", "threshold": 0.6}'
```

The agent holds everything about routing that is not the model: which teams
exist, where their work begins and ends, and how sure the model must be before a
conversation is routed rather than withheld. The model only scores the
options it is given.

One honest caveat about the chapter's own round. `src/decide.py` builds the
typed question itself and sends it, which the agent passes straight through to
the model, so the threshold on a `make decide` run is applied by the client and
the agent's policy branch is not what decided anything. That is deliberate: the
same question goes to all three backends, so the local, HTTP and AgentCore
numbers compare. A caller that wants the policy sends `{"text": ...}` and gets
`routed` back, which is what the curl above does. That division is the point of the split, and it is why the
agent's IAM policy allows exactly one action, `sagemaker:InvokeEndpoint` on one
endpoint.

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
probability has to be a finite number in [0,1], with the categories totaling
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

Confidence is normalized max-probability, `(N * p_max - 1) / (N - 1)`. That puts
the number on the same scale whatever the option count, uniform at 0 and one-hot
at 1, so a threshold is at least comparable across team lists. It does not make
one threshold equally reliable across them: how often 0.6 is right depends on the
distribution the model produces for that option set, and the model card is
explicit that calibration is fitted per primitive and should be measured on your
own traffic. Changing the team list means measuring the threshold again.

Two things worth knowing before building on it:

- The checkpoint is Apache-2.0 and is two downloads, not one: the adapter and head
  are about 0.235 GB, and the Qwen3.5-2B-Base torso underneath is another 4.571 GB,
  fetched on first load. Both downloads are baked into the serving image during the build.
- `strands-decider` is version 0.3.0 and its own description calls it an
  experimental typed classifier. Pin it, and expect the API to move.

Latency scales with the length of the state, so the sub-200 ms figures quoted for
short tasks are not what a multi-turn transcript costs; see the numbers below.

## Two routers, one question

Both routers answer the same question about the same 320 conversations, briefed
from the same definitions in `src/categories.py`. Measured on 2026-11-07, the
decision model on an M3 laptop and the zero-shot LLM through the local batch shim:

| router | accuracy | macro-F1 | Brier | coverage | median |
| --- | --- | --- | --- | --- | --- |
| decision model, deployed (AgentCore + SageMaker, g5) | 0.669 | 0.670 | 0.515 | 1.000 | 397 ms |
| decision model, local (M3, MPS) | 0.666 | 0.666 | 0.515 | 1.000 | 1908 ms |
| Claude Haiku 4.5, Bedrock batch | 0.69 | | | | |
| zero-shot LLM, local shim (Qwen3 0.6b) | 0.156 | 0.059 | 1.687 | 0.997 | ~1170 ms |

The first two rows are the chapter's claim about local-first, measured rather
than asserted: 0.666 on a laptop and 0.669 deployed, on the same 320
conversations. The local round said what the cloud round would say. The GPU
bought latency, not accuracy, and it bought a lot of it: 397 ms against 1908 ms,
138 seconds for the set against 639. Both still run reference PyTorch kernels,
since neither `causal_conv1d` nor `flash-linear-attention` is installed, so
there is headroom in that number.

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
what remains improves from 0.496 to 0.409. Coverage of 0.800 means one
conversation in five is withheld, which is the price of the better Brier on the
other four.

Withheld is all it is. The record carries no prediction and the reason, so the
scorer counts it as uncovered and `route.py` publishes nothing for it. Nothing
here delivers it to a review queue; wiring those to a team is the obvious next
step and the chapter does not take it.

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
- `agent/main.py`: the routing agent, which owns the policy and asks the model;
  the same file locally and on AgentCore Runtime
- `model/Dockerfile`: the decision model behind a SageMaker endpoint, serving the
  same `/invocations` and `/ping` contract with no bespoke inference code
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
- `tests/test_evaluate.py`: the scoring math, and what a missing prediction
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
  AgentCore Runtime (the routing agent, serverless), Amazon SageMaker (the
  decision model, on a GPU endpoint), Amazon SNS, Amazon S3
- **Local stand-ins:** a from-source Bedrock batch shim over S3Proxy + Ollama, a
  from-source SNS shim (file inboxes), S3Proxy (S3). The decision model has no
  stand-in at all: `agent/main.py` serves the AgentCore Runtime contract itself,
  so the local round runs the file that deploys, and `make sm-local` runs the
  CPU image built from the deployment Dockerfile through SageMaker local mode.
  It checks the entrypoint and API using a different base image and PyTorch build. `make decide` needs no container at all, where the batch path needs
  S3Proxy for its manifests.

## Run it on AWS

```
make -C aws iam            # ch10-user (privileged identity), then assume it via a profile
make -C aws deploy         # the batch bucket + the bedrock.amazonaws.com service role
make -C aws classify       # a real CreateModelInvocationJob (tens of minutes)
make -C aws route          # publish to real SNS topics
make -C aws eval
make -C aws teardown
```

### The decision model and its agent

Off by default, because it provisions a GPU endpoint that bills while it is up.

```
make -C aws decider-round      # the steps below, in order
```

There are two deployables and they are built differently, which is the thing
worth noticing:

```
make -C aws deploy WITH_DECIDER=true   # repository, bucket, CodeBuild, roles
make -C aws image                      # the model image: x86_64, CUDA torch, weights baked in
make -C aws agent                      # the agent: an arm64 zip, no image at all
make -C aws deploy WITH_DECIDER=true DECIDER_IMAGE=<repo>:latest AGENT_KEY=agent.zip
make -C aws decide                     # route the dataset through the agent
```

The model needs a container because it carries CUDA torch and 4.5 GB of weights,
and the weights are baked into the image rather than downloaded on demand
because SageMaker gives `/invocations` 60 seconds to answer. The agent needs no
container: it is small enough for direct code deployment, a zip under 250 MB,
built for `aarch64` because AgentCore Runtime is arm64 only.

Two names in the template cannot take hyphens, and the validation error does not
say so: the runtime is `ch10_router`.

Expect the endpoint to be the slow part. The image is large, so the first deploy
waits on a CodeBuild that pulls the weights, and
`ContainerStartupHealthCheckTimeoutInSeconds` is raised to 900 for the same
reason. Verified on 2026-11-07: InService in 420 seconds, then 320
conversations routed in 138.

### What the endpoint took seven attempts to teach

Each of these failed a deploy, and none of them is about the model.

- **The base image decides which torch you get.** An image with no torch means
  pip installs the default wheel, built against a newer CUDA than SageMaker's
  GPU hosts carry, and the container dies in `torch._C._cuda_init()` before it
  can answer `/ping`. SageMaker Distribution ships torch 2.8.0 matched to its
  own hardware. The PyTorch inference DLC would be the usual choice and cannot
  be used here: it stops at torch 2.6.0, below what this model needs. Check a
  dependency floor against the images your platform publishes before designing
  around one.
- **SageMaker Distribution runs its own inference server.** It expects a handler
  at `/opt/ml/model/inference.py` and ignores a CMD entirely. Either write that
  handler, or take the `ENTRYPOINT`, which is what `model/entrypoint.sh` does.
- **`docker run <image> serve`.** That argument replaces a CMD, so a plain
  `docker run <image>` passes while the endpoint fails. `make sm-local` starts
  the container the way the service does, which is the only local check that
  catches it.
- **A mutable tag is not a deploy.** `<repo>:latest` is an unchanged
  CloudFormation parameter, so the model is never replaced and the endpoint
  keeps serving the old digest while the deploy reports success. The image
  carries the build's hash as its tag for that reason, and the agent zip carries
  its own hash as its key for the same one.
- **GPU capacity runs out**, and a single `InstanceType` answers
  `InsufficientInstanceCapacity` only after twenty minutes. `InstancePools`
  walks several by priority. Only list types whose drivers match the build:
  g4dn has capacity when others do not, carries a CUDA 11.4 driver, and so
  turns a fast failure into a slow one.
