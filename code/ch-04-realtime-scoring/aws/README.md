# The decision service on AWS

The same container behind an Application Load Balancer on ECS Fargate, with
the decision log in DynamoDB. The service deploys at desired count 0, so
the stack costs nothing until you scale it up.

## Before the first deploy

The deploying user needs EC2 (the VPC), ELB, ECS, ECR, and CodeBuild
grants; they are collected in `iam/deploy.json`. Attach once, on the
chapter's IAM group:

```
aws iam create-group --group-name book-ch04
aws iam put-group-policy --group-name book-ch04 \
  --policy-name CreditBookCh4Deploy --policy-document file://iam/deploy.json
aws iam add-user-to-group --group-name book-ch04 --user-name <you>
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
