# Governance, security, and safe rollout

> **Scaffold in progress.** This chapter is being built on `ch09-ch12`; the
> layout below is the plan, not yet a verified run. Sections fill in as each
> piece is written and tested locally, then on AWS.

What makes the systems from the earlier chapters audit-ready and safe to change:
the paperwork auditors need, the least-privilege the security review needs, and
a new model shipped without a redeploy.

## What it builds

- **Model cards**, the **model registry**, and **lineage** — the record an
  auditor reads to see what a model is, how it was built, and what it was
  trained on.
- **Least-privilege IAM** for the security review.
- A new model shipped with **champion/challenger rollout** and an **A/B test**,
  flipped through **AppConfig feature flags** with no redeploy.

AppConfig runs locally from source — the flag store and rule evaluator carried
over from `appconfig_agent` and the ION-format `appconfig-local` work — so the
same flag flip drives the rollout locally and on AWS.

## Planned layout

- `src/`: the champion/challenger router and the A/B assignment, tested here
  first
- `local/`: compose stack — the from-source AppConfig flag store + rule evaluator
- `aws/template.yaml`: SAM — model registry / model cards, AppConfig application +
  environment + feature-flag configuration, scoped IAM roles
- `diagrams/`: cloud architecture and its local mirror

## AWS services and local stand-ins

- **AWS services:** Amazon SageMaker Model Cards, SageMaker Model Registry,
  SageMaker ML Lineage, AWS AppConfig, AWS IAM
- **Local stand-ins:** an AppConfig flag store + rule evaluator composed from
  source (ION-format configuration)
