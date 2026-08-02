# The decision service on AWS

The same container behind an Application Load Balancer on ECS Fargate, with
the decision log in DynamoDB. The service deploys at desired count 0, so
the stack costs nothing until you scale it up.

## Before the first deploy

This chapter deploys as `ch04-user`, a role carrying only the grants in
`iam/deploy.json` (EC2/VPC, ELB, ECS, ECR, DynamoDB, CodeBuild, plus the
CloudFormation and scoped IAM needed to create the stack). Bootstrap the role
once and assume it via a profile — see
[`code/README.md`](../../README.md) (`make iam` in aws/). Then deploy under
that profile:

```
make deploy AWS_PROFILE=ch04
```

## Run it, in order

```
make deploy      # the stack: VPC, ALB, cluster, table, ECR, CodeBuild
make image       # build the service image in the cloud
make scale-up    # one Fargate task behind the ALB
make decide      # one application through the real endpoint
make scale-down  # stop paying for the task
make teardown    # remove the stack
```

Still to come: the SageMaker endpoint behind the service, Step Functions,
and the burst load test.
