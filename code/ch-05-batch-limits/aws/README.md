# The batch limit run on AWS

The stack carries the pipeline's bucket, the step image built in the cloud,
the role the pipeline runs as, the EventBridge schedule that starts it, and
an SNS topic that hears about failures. The pipeline definition is the same
`src/pipeline/pipeline.py` that ran locally, registered with
`PIPELINE_MODE=aws`. The warehouse is the chapter's own Redshift Serverless in
its **own self-contained VPC** (subnets, route table, and the VPC endpoints the
in-VPC jobs need — all declared in `template.yaml`, so there is nothing to look
up). It is created only when `DeployWarehouse=true` (it bills per second while
active, so add it for the run and set it back to `false`, or tear down, the same
day). The pipeline's jobs run in that VPC and reach the private warehouse with
`psycopg2` (`sslmode=require`); the one-time seed loads it over the Redshift Data
API. `make pipeline` reads the endpoint, password, subnets, and job security
group from the stack outputs, so no values are passed by hand.

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
make deploy      # the whole stack: self-contained VPC, Redshift, ECR, CodeBuild, schedule, alerts
make image       # build the step image in the cloud
make seed        # load the warehouse over the Data API (upload CSV + COPY FROM S3)
make pipeline    # register + run; DSN and VpcConfig come from the stack outputs
make run-now     # a second execution now (the schedule owns the rest)
make teardown    # remove the pipeline and the whole stack when the run is done
```

`make deploy` is one `sam deploy` — the whole network is in the template, so there
is nothing to look up. The warehouse bills per second, so tear down the same day.
Add an alert email with `sam deploy --parameter-overrides DeployWarehouse=true
AlertEmail=you@example.com`.
