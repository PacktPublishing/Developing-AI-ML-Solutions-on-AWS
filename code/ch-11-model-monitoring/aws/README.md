# The monitoring run on AWS

The chapter's drift monitors run the same code locally and on AWS: the model's
own SHAP values for feature attribution, PSI for score and feature drift, and
the standardized bias metrics in `src/bias.py`. On AWS they run inside the
pipeline's Processing step rather than as a separate job.

SageMaker Clarify is closed to new customers. AWS's [replacement
guidance](https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-availability-change.html)
is to compute the bias metrics directly, because they are standardized
arithmetic rather than a service, and to use SHAP, which is the engine Clarify
was built on, so that is what this chapter does. `tests/test_bias.py` holds each
metric to `smclarify`, AWS's own open-source implementation, so the numbers are
checked against the reference instead of being asserted to match it.

The stack (`template.yaml`) is deliberately small: the ECR repository for the
scorecard's train/serve image, the role the jobs and the endpoint run as, the
bucket they read and write, and the CloudWatch alarms that watch drift.
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
make round       # the four below, in order
make deploy      # the repo, the role, the bucket and the drift alarms
make image       # build the train/serve image for linux/amd64 and push to ECR
make pipeline    # Batch Transform to score, then Processing to monitor
make endpoint    # a live endpoint with Data Capture, and 6000 rows through it
make teardown    # empty the S3 prefix and delete the stack when the run is done
```

## What the round produced

Verified cold on 2026-11-08 as `ch11-user`, from an account with no stack in it,
and torn down after: no stack, no endpoint.

| step | result |
| --- | --- |
| `pipeline` | `score-current` Succeeded, `monitor-drift` Succeeded |
| `endpoint` | InService, 6000 rows sent, capture landing in about 45 seconds |
| monitor over the capture | `score_psi=0.7488`, `attribution_ndcg=0.986`, 4 violations |

PSI's major-shift threshold is 0.25, so a score PSI of 0.7488 is the drift the
chapter exists to catch, and the alarms fired on it along with
`debt_to_income`, `bureau_score` and `credit_utilization`.

The bias metrics came out the same from the batch on a laptop and from the
endpoint's own captured traffic:

| facet | CI | DI | DPPL |
| --- | --- | --- | --- |
| `employment_status=unemployed` (357 of 6000) | +0.8810 | +1.1200 | -0.0555 |
| `home_ownership=RENT` (1827 of 6000) | +0.3910 | +1.2019 | -0.0886 |

Both facets receive a higher rate of predicted defaults than the rest of the
population: DPPL is negative and DI is above 1, which is the same statement
twice. Whether that is the model or the data is a question the metrics raise
rather than answer.

Only these three are reported for a live endpoint. The other seven need the
outcome of each decision, and captured traffic holds the request and the score
and nothing about what happened next.

CloudWatch has all ten under `ch11/monitoring`, as `bias_<metric>` with a
`Facet` dimension, so they alarm and trend like any other metric.

## A reader's account needs less than you would think

`iam/deploy.json` is 17 statements of explicit actions, scoped to `ch11-*`, with
no service wildcards. Narrowing it from `sagemaker:*`, `cloudformation:*`,
`ecr:*`, `s3:*` and `logs:*` is what surfaced two defects a wildcard hides: the
artifacts were going to `sagemaker-<region>-<account>`, the account's default
bucket that no chapter owns, and the execution role was found by listing every
role in the account and matching a name that only exists where someone has
opened Studio.
