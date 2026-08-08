# Self-service analytics with Claude Code on Bedrock

Analysts ask questions in plain English; Claude Code writes read-only SQL, runs
it against the warehouse through the Redshift Data API, and answers with a
table, a short summary, and the tables it touched. The terminal runs in the
browser through ttyd, the model runs on Amazon Bedrock, and the container
never holds a database password: locally the Data API is a shim over
redshift-local, on AWS it is the real API under an IAM task role.

## Run it locally

```
make llm                                  # serve gpt-oss-20b on vllm-metal
make local                                # fully local, no account, no login
AWS_PROFILE=<profile> make local-bedrock  # real Bedrock, Fargate-style creds
make local-stop                           # tear either down
```

`make local` is local end to end: Claude Code drives gpt-oss-20b through
vLLM's native Anthropic API on the vllm-metal plugin (MLX on Apple Silicon),
the warehouse is redshift-local behind the from-source Data API shim, and
the terminal starts straight at the prompt because the credentials are
ambient, exactly as on the Fargate task. Budget memory honestly: the model
wants 16 GB to itself, and running it beside the stack and a browser needs
more than 32 GB to stay interactive; below that, `make local-bedrock` is
the fast local session.
`make local-bedrock` is the pre-deploy parity check, with credentials from
the `amazon-ecs-local-container-endpoints` sidecar, the task-metadata path
the task role fills on Fargate. `make local-claude` swaps in your own Claude
account for the fastest local sessions.

Open http://localhost:7681 and ask: "Which state has the highest average
credit score?" The seeded warehouse is the credit mart from the data
engineering chapter.

## Layout

- `Dockerfile`: Claude Code + ttyd + uv on Ubuntu, non-root, `noshell` blocks
  shell escape from the terminal
- `docker-entrypoint.sh`: starts ttyd with the restricted launcher
- `launch-claude`: per-user Claude state, Bedrock auth via the task role
- `CLAUDE.md`: the assistant's contract — read-only SQL, no secrets, dialect
  gotchas, verify-before-filter, zero results are a red flag
- `skills/redshift-data-api/`: the SQL tool — Data API helper with timeout,
  row caps, and result masking; `SKILL.md` teaches Claude to use it
- `config/pii_columns.yaml`: single source of truth for masked columns
- `config/provision.sh`: the warehouse side of the security model — the
  read-only bi_analyst account (PASSWORD DISABLE, IAM-only) and engine-side
  masking policies generated from the same yaml, so the two layers cannot
  drift; cloud-only, since redshift-local has no native masking
- `local/`: compose stack — redshift-local, a from-source Data API shim, the
  assistant; `seed/` loads the credit mart
- `aws/template.yaml`: SAM — Redshift Serverless, ECS Fargate service
  (desired count 0 at rest), scoped task role, ECR, logs

## Cost posture

Everything at rest costs nothing: the Fargate service parks at desired count
0, Redshift Serverless bills per second only while a query runs, and Bedrock
bills per token. The chapter's cloud round is deploy, ask, tear down.
