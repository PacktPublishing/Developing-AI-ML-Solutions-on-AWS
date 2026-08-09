# Governance, security, and safe rollout

What makes the systems from the earlier chapters audit-ready and safe to change: the
paperwork auditors need, the least-privilege the security review needs, and a new model
shipped without a redeploy.

## What it builds

- A new model shipped with **champion/challenger rollout**, flipped through **AppConfig
  feature flags** with no redeploy. A FastAPI gateway routes each application to the
  champion (chapter 2's WOE scorecard) or the challenger (its monotone XGBoost) by the
  `challenger_rollout` flag; the flag's `split` rule buckets each loan deterministically,
  so widening the rollout is a flag edit and no loan flips back.
- **Model cards**, a **model registry**, and **lineage** — the record an auditor reads to
  see what a model is, how it was built, and what it was trained on.
- **Least-privilege IAM** for the security review.

AppConfig runs locally from source: the rule evaluator (S-expression rules and the
FNV-1a `split` bucketing) is byte-identical to the reverse-engineered AppConfig agent, so
a loan buckets the same locally and under the real agent on AWS — the rollout has true
local/cloud parity.

## Run it locally

Train chapter 2's two models first (`make train` and `make train-challenger` in
`../ch-02-mlops`); the gateway routes to those serving images.

```
make up                 # champion + challenger + the flag-routing gateway (:8080)
make apps               # a batch of applications (chapter 2's held-out set)
make score              # route the batch; see the champion/challenger split
make flip PCT=50        # widen the rollout with nothing restarted
make score              # the split has shifted
make govern             # write the model cards, registry, and lineage
make down
```

Verified end to end against the real models: 20% → champion 234 / challenger 66, flip
50% → 158 / 142, flip 100% → all challenger, roll back 20% → 234 / 66 (deterministic per
loan). The challenger genuinely beats the incumbent on the held-out set (AUC 0.8825 vs
0.8625), which is why it ships behind a flag rather than all at once.

## Layout

- `src/appconfig/`: the from-source AppConfig seam — the rule evaluator and the
  `get_appconfig()` client (local flag store, or the AppConfig agent on AWS)
- `src/router.py`, `src/models.py`: the champion/challenger router over the shared
  `/invocations` contract
- `app/main.py`: the FastAPI rollout gateway (uvicorn locally, a Fargate service on AWS)
- `src/governance.py`, `governance/models.json`: the model card / registry / lineage
  generator and its inputs
- `local/`: the compose stack and the feature-flag document
- `aws/`: the AWS round (real AppConfig + the ECS agent, SageMaker Model Cards / Registry
  / ML Lineage, scoped IAM) — next
- `diagrams/`: the local stack and its cloud mirror

## AWS services and local stand-ins

- **AWS services:** AWS AppConfig (+ the ECS/Fargate agent sidecar), Amazon SageMaker
  Model Cards, SageMaker Model Registry, SageMaker ML Lineage, AWS Fargate, AWS IAM
- **Local stand-ins:** the AppConfig flag store + rule evaluator composed from source
