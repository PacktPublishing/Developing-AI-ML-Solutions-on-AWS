# The batch limit run on AWS

The stack carries the pipeline's bucket, the step image built in the cloud,
the role the pipeline runs as, the EventBridge schedule that starts it, and
an SNS topic that hears about failures. The pipeline definition is the same
`pipeline/pipeline.py` that ran locally, registered with
`PIPELINE_MODE=aws`. The warehouse is the chapter's own Redshift Serverless
(`redshift-serverless.yaml`, deployed separately because its lifetime is
shorter — it bills per second while active); pass its endpoint as
`WAREHOUSE_DSN`.

## Before the first deploy

This chapter deploys as `ch05-user`, a role carrying only the grants in
`iam/deploy.json` (SageMaker Pipelines, ECR, CodeBuild, EventBridge, SNS,
Redshift read, plus the CloudFormation and scoped IAM needed to create the
stack). Bootstrap the role once and assume it via a profile — see
[`code/README.md`](../../README.md) (`code/setup-users.sh`). Then deploy under
that profile:

```
make deploy AWS_PROFILE=ch05
```

## Run it, in order

```
aws cloudformation deploy --template-file redshift-serverless.yaml \
  --stack-name ch05-redshift --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides VpcId=... SubnetIds=... AllowedCidr=x.x.x.x/32 \
  AdminPassword=...   # the warehouse; delete it the same day
make deploy      # bucket, ECR, CodeBuild, roles, schedule, alerts
make image       # build the step image in the cloud
make pipeline WAREHOUSE_DSN=postgresql://...   # register the pipeline
make run-now     # one execution now; the schedule owns the rest
make teardown    # remove the pipeline and the stack
```

Deploy with an email to hear about failures:

```
sam deploy --parameter-overrides AlertEmail=you@example.com
```
