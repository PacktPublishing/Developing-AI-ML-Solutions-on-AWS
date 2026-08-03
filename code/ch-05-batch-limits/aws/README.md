# The batch limit run on AWS

The stack carries the pipeline's bucket, the step image built in the cloud,
the role the pipeline runs as, the EventBridge schedule that starts it, and
an SNS topic that hears about failures. The pipeline definition is the same
`src/pipeline/pipeline.py` that ran locally, registered with
`PIPELINE_MODE=aws`. The warehouse is the chapter's own Redshift Serverless,
declared in this `template.yaml` behind the `HasWarehouse` condition — it is
created only when you pass `VpcId` (it bills per second while active, so add
it for the run and update the stack back without `VpcId` the same day). Pass
its endpoint as `WAREHOUSE_DSN`.

## Before the first deploy

This chapter deploys as `ch05-user`, a role carrying only the grants in
`iam/deploy.json` (SageMaker Pipelines, ECR, CodeBuild, EventBridge, SNS,
Redshift read, plus the CloudFormation and scoped IAM needed to create the
stack). Bootstrap the role once and assume it via a profile — see
[`code/README.md`](../../README.md) (`make iam` in aws/). Then deploy under
that profile:

```
make deploy AWS_PROFILE=ch05
```

## Run it, in order

```
make deploy      # bucket, ECR, CodeBuild, roles, schedule, alerts (no warehouse)
make image       # build the step image in the cloud

# add the warehouse for the run: VpcId turns on the Redshift Serverless
# resources (admin password is generated into Secrets Manager). Update the
# stack back without VpcId the same day to stop the per-second billing.
sam deploy --parameter-overrides \
  VpcId=vpc-xxxx WarehouseSubnetIds=subnet-a,subnet-b,subnet-c AllowedCidr=x.x.x.x/32

make pipeline WAREHOUSE_DSN=postgresql://...   # register the pipeline
make run-now     # one execution now; the schedule owns the rest
make teardown    # remove the pipeline and the stack
```

Deploy with an email to hear about failures:

```
sam deploy --parameter-overrides AlertEmail=you@example.com
```
