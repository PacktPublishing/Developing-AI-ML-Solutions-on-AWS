# Monitoring models in production

> **Scaffold in progress.** This chapter is being built on `ch09-ch12`; the
> layout below is the plan, not yet a verified run. Sections fill in as each
> piece is written and tested locally, then on AWS.

What keeps a bad model from making bad decisions at scale: online drift
monitoring of the scores and the features that drive them, drift in the
scorecard's own feature contributions, and the managed AWS monitors, with
CloudWatch alerting on top.

## What it builds

- Online **Population Stability Index (PSI)** monitoring of scores and features.
- **SHAP-based** detection of drift in a scorecard's feature contributions —
  built on **CatBoost**, which exposes native `feature_importances_`.
- **SageMaker Model Monitor** and **Clarify** for data and concept drift,
  including [feature-attribution drift](https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-model-monitor-feature-attribution-drift.html).
- **CloudWatch alerting** that fires before bad decisions accumulate.

Like the other SageMaker chapters, it runs in **SageMaker local mode** first
(V3 SDK) and then on AWS with the same code.

## Planned layout

- `src/`: the PSI and SHAP-drift code, tested here first
- `local/`: compose stack — SageMaker Monitor/Clarify and CloudWatch shims
  composed from source (modelled on the reference in `localstack-pro`)
- `aws/template.yaml`: SAM — Model Monitor + Clarify schedules, CloudWatch alarms
- `notebooks/`: the PSI + SHAP-drift narrative over the tested `src/`
- `diagrams/`: cloud architecture and its local mirror

## AWS services and local stand-ins

- **AWS services:** Amazon SageMaker Model Monitor, SageMaker Clarify, Amazon
  CloudWatch
- **Local stand-ins:** SageMaker Monitor/Clarify and CloudWatch shims composed
  from source; SHAP + CatBoost run directly
