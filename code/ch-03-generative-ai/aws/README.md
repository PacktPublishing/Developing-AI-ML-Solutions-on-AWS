# On AWS

The chapter runs on your machine first. This folder takes it to real AWS: a smoke
check of the Bedrock models, and a SAM stack in `assistant-api/` that deploys the
whole assistant. Bedrock runs on real AWS in every chapter run too, since there
is no local Bedrock; the offline path swaps in Ollama. `make smoke` tees its
output to `/tmp` with the date.

```
make check-account   # which account the runs use
make smoke           # Titan embeddings + Qwen3 Converse on real Bedrock
```

## Deploy the assistant

`assistant-api/` is a SAM application: Amazon RDS for PostgreSQL with pgvector as
the store and a Lambda, behind a function URL, that answers a question the same
way the chapter does. The Lambda seeds itself on its first call, so one deploy
leaves a working endpoint. RDS `db.t3.micro` is free-tier. Aurora is the scale-up,
but Aurora on a free-plan account needs an express configuration that
CloudFormation does not yet expose, so the template uses plain RDS.

```
cd assistant-api
make deploy      # sync the knowledge base, sam build, then sam deploy
make teardown    # when done
```

The knowledge base has one source of truth, `corpus_data.py` at the chapter root.
A Lambda package has to be self-contained, so `make sync` copies that file into
`src/` at build time rather than keeping a second copy in the repository. The
deployed assistant therefore holds the same sector profiles and credit policy as
the local run, and the retrieval and prompt match the chapter's ask path.

To apply the guardrail on the deployed assistant, create it in the chapter
(`make guardrail`, which saves the id and version to `data/guardrail.json`) and
pass both to the stack:

```
sam deploy --parameter-overrides GuardrailId=<id> GuardrailVersion=<version>
```

With both set, the guardrail runs inside the Converse call with the retrieved
passages as the grounding source, the same way the chapter applies it. Left
empty, the stack deploys and answers without one.

Deploying creates RDS, a Lambda, a VPC, and IAM roles, so run it with a principal
that can create those, an administrator or a power user with IAM. The specific
permissions are collected in `iam/`.

## IAM permissions

Runtime permissions the assistant needs and the deploy permissions the stack
needs live in `iam/`. Scope every resource to your account and region in
production.

| Action | Why |
|---|---|
| `bedrock:InvokeModel` | Titan Text Embeddings v2 |
| `bedrock:InvokeModel` | Qwen3 generation (the Converse API is authorized by this action) |
| `bedrock:ApplyGuardrail` | the guardrail on the Converse call |
| `bedrock:CreateGuardrail` and the guardrail lifecycle | manage the guardrail |
| CloudFormation, S3 (SAM bucket), EC2 (VPC), RDS, Secrets Manager, Lambda, IAM (the function role) | deploy the stack (`iam/deploy.json`) |

Qwen3 is served on-demand, so a single `bedrock:InvokeModel` on the foundation
model is enough. A Claude model would instead need a cross-region inference
profile, with the permission covering both the profile and the underlying model.
