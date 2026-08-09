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
- **Model cards**, a **model registry**, and **lineage**: the record an auditor reads to
  see what a model is, how it was built, and what it was trained on.
- **Least-privilege IAM** for the security review.

The rollout runs the **real AWS AppConfig agent**: the same published agent image is the
sidecar locally (in local development mode, reading an Ion feature-flag file) and on ECS
(fetching from AppConfig), so the `split` buckets every loan identically in both worlds by
construction. The from-source rule evaluator in `src/appconfig/` is a reference for the
non-split logic and for running without Docker; its `split` is close but not bit-identical
to the agent's, which is why the rollout decision goes through the agent.

## Run it locally

Train chapter 2's two models first (`make train` and `make train-challenger` in
`../ch-02-mlops`); the gateway routes to those serving images.

```
make up                 # the AppConfig agent + champion + challenger + gateway (:8080)
make apps               # a batch of applications (chapter 2's held-out set)
make score              # route the batch; see the champion/challenger split
make flip PCT=50        # widen the rollout; the gateway and models never restart
make score              # the split has shifted
make govern             # write the model cards, registry, and lineage
make down
```

Verified end to end against the real models, routed by the real agent: 20% gives champion
238 / challenger 62, flip 50% gives 140 / 160, flip 100% gives all challenger, roll back
20% gives 238 / 62 (deterministic per loan). The challenger beats the incumbent on the
held-out set (AUC 0.8825 vs 0.8625), which is why it ships behind a flag rather than all
at once.

## Layout

- `src/appconfig/`: the from-source AppConfig seam (the rule evaluator and the
  `get_appconfig()` client: local flag store, or the AppConfig agent on AWS)
- `src/router.py`, `src/models.py`: the champion/challenger router over the shared
  `/invocations` contract
- `app/main.py`: the FastAPI rollout gateway (uvicorn locally, a Fargate service on AWS)
- `src/governance.py`, `governance/models.json`: the model card / registry / lineage
  generator and its inputs
- `local/`: the compose stack and the feature-flag document
- `aws/`: the AWS round, next (real AppConfig + the ECS agent, SageMaker Model Cards /
  Registry / ML Lineage, scoped IAM)
- `diagrams/`: the local stack and its cloud mirror

## AWS services and local stand-ins

- **AWS services:** AWS AppConfig (+ the ECS/Fargate agent sidecar), Amazon SageMaker
  Model Cards, SageMaker Model Registry, SageMaker ML Lineage, AWS Fargate, AWS IAM
- **Local stand-ins:** the AppConfig flag store + rule evaluator composed from source
