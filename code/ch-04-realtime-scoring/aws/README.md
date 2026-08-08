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

## Where the two paths differ (and why)

- **The identity, side by side.** On ECS the pod's permissions come from the
  `TaskRole` in `template.yaml`; on EKS they come from the `gateway` service account,
  whose IAM role is bound with `iam.withOIDC: true` in `eks/create-cluster.yaml`.
  That contrast is the point: ECS hands a task its role directly, while Kubernetes
  needs a federation bridge (IRSA, an OIDC issuer). Note that EKS **Pod Identity**
  (`podIdentityAssociations`) is now the closer parallel to a task role — an agent
  vends credentials, like the ECS task-metadata endpoint, with no OIDC issuer. This
  chapter uses IRSA to show the mechanism; Pod Identity is the current default.

- **Reaching the pod on EKS is a port-forward, on purpose.** `eksctl` does not
  install the AWS Load Balancer Controller, so an Ingress would never get an address.
  `make decide COMPUTE=eks` uses a `kubectl port-forward`; a real ingress needs the
  controller (its own IAM policy and service account), out of scope here. The ECS
  path has the ALB the stack builds.

- **Logging is asymmetric.** ECS uses the `awslogs` driver, so decisions land in
  CloudWatch. EKS Fargate has no node and no log agent unless you create the
  `aws-observability` namespace with the Fluent Bit config; this chapter does not, so
  on EKS use `kubectl logs deploy/gateway` — nothing reaches CloudWatch.

- **Tear down before switching `COMPUTE`.** Going from a running ECS stack straight
  to `COMPUTE=eks` asks CloudFormation to delete the ECS service and its VPC in one
  changeset; ENI cleanup on an active Fargate task can stall the VPC delete for ~20
  minutes. Run `make teardown` first, then deploy the other backend.
