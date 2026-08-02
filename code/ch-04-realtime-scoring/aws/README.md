# The decision service on AWS

The gateway on ECS Fargate (or EKS) behind an Application Load Balancer,
orchestrating two SageMaker endpoints, with the decision log in DynamoDB. The
gateway deploys at desired count 0, so the stack costs nothing until you scale it
up; the endpoints are serverless and scale to zero.

## Before the first deploy

`make iam` bootstraps two roles: `ch04-user`, the deploy identity carrying only
the grants in `iam/deploy.json`, and `ch04-sagemaker`, the execution role the
endpoints run as. Assume `ch04-user` through a profile — see
[`code/README.md`](../../README.md) — and pass the `ch04-sagemaker` ARN it prints
as `SM_ROLE`.

## Run it, in order

The models first: build and push each serving image (linux/amd64, the Docker v2
manifest SageMaker requires), then deploy both as serverless endpoints.

```
make push-models                         # build + push scorecard, challenger to ECR
make endpoints SM_ROLE=<ch04-sagemaker>  # deploy both as serverless SageMaker endpoints
```

`make endpoints` runs `../src/deploy.py MODE=cloud`, the same script `make sm-local`
runs with `MODE=local` — one deploy, the mode is the only difference.

Then the gateway. Pick where its pods run with `COMPUTE` — `ecs` (default) or `eks`:

```
make deploy                # gateway on ECS Fargate behind an ALB (default)
make deploy COMPUTE=eks    # gateway on an eksctl EKS Fargate cluster
make decide                # one application through the endpoint
make decide COMPUTE=eks    #   (the ALB for ecs; a kubectl port-forward for eks)
make teardown              # remove the gateway stack and the endpoints
make teardown COMPUTE=eks  #   (also deletes the EKS cluster)
```

The gateway invokes the two endpoints with `sagemaker-runtime`, averages their PDs,
and writes the decision to DynamoDB; `COMPUTE` only changes where the gateway pod
runs (its ECS task role and its EKS service account both carry `InvokeEndpoint`).
`make scale-down` stops the ECS task without tearing the stack down.
`make endpoints-teardown` removes the endpoints without touching the gateway. EKS's
control plane bills hourly — create, run, delete.
