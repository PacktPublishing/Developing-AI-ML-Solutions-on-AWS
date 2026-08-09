# The monitoring run on AWS

The chapter's drift monitors run locally against the model's own SHAP values; on
AWS the same feature-attribution check runs as a real **SageMaker Clarify**
explainability job. The stack (`template.yaml`) is deliberately small: the ECR
repository for the scorecard's train/serve image, and the CloudWatch alarm that
watches feature-attribution drift. The Clarify jobs themselves have no
CloudFormation resource, so `clarify_monitor.py` launches them with the SDK.

`make clarify` runs two explainability jobs — one on the reference batch, one on the
drifted current batch. Each stands up a shadow endpoint from the scorecard model,
computes global mean-absolute SHAP with `SHAPConfig`, and writes an `analysis.json`.
The launcher reads the two attribution rankings, scores the **NDCG** between them
(the same measure `src/monitor.py` uses, and the one Clarify itself reports),
publishes it to CloudWatch, and reads back the `ch11-attribution-drift` alarm.
Clarify raises a flag when that NDCG falls below 0.90.

> **Model Monitor's *scheduled* service is closed to new accounts.** SageMaker
> Clarify still runs as an on-demand processing job, which is what this chapter
> uses: the same explainability analysis a schedule would run, launched when you
> want it. The scheduled `CreateMonitoringSchedule` wrapper is the piece that is no
> longer open.

## Before the first deploy

This chapter deploys as `ch11-user`, a role carrying only the grants in
`iam/deploy.json` (SageMaker, ECR, S3 on the SageMaker buckets, CloudWatch, plus
the CloudFormation and PassRole needed to create the stack and run Clarify as the
SageMaker execution role). Bootstrap the role once as the account admin and assume
it via a profile — see [`code/README.md`](../../README.md) (`make iam` in aws/).

## Run it, in order

First train the scorecard locally (`make data && make train` in the chapter root):
the AWS round packages that model and monitors it.

```
make deploy      # the ECR repo + the feature-attribution-drift alarm
make image       # build the train/serve image for linux/amd64 and push to ECR
make package     # upload the trained scorecard as a SageMaker model.tar.gz
make clarify     # two Clarify explainability jobs, the NDCG drift, and the alarm
make teardown    # empty the S3 prefix and delete the stack when the run is done
```
