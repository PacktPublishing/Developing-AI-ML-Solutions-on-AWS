# Real-time credit scoring behind an endpoint

The running example for the real-time chapter. A lender's core product moment
is a loan application answered while the applicant waits: the service takes
the application, the model gives a probability of default, and the policy
turns that — together with hard rules — into approve, refer, or decline,
with the reason codes an adverse-action notice is built from. The decision,
not the score, is the product.

The service is FastAPI with typed contracts, and it knows nothing about
AWS: it runs anywhere a container does. The same image runs under uvicorn
on a laptop and on ECS Fargate behind a load balancer, and nowhere does the
code know which.

## Run it

```
make up        # the service and its decision log
make decide    # a clean application: APPROVE
make refer     # utilization in the gray band: REFER, with the reason
make decline   # failed KYC: DECLINE, whatever the score says
make log       # the audit trail in DynamoDB
make down      # stop and clean
```

## Local vs AWS

| In this directory | On AWS |
| --- | --- |
| the service container on :8080 | the same image on ECS Fargate behind an ALB |
| `/health` answered to nobody | the ALB target-group health check |
| DynamoDB Local decision log | DynamoDB |

## The same image on Kubernetes

The portability claim, proven on a second orchestrator: `make kube-up`
starts a k3d cluster (k3s in Docker), imports the image, and applies
`k8s/decision-service.yaml` — a Deployment with `/health` wired to the
readiness and liveness probes (the target group's job), a Service, and an
Ingress (the ALB's job). `make kube-decide` answers through it, and
`make kube-down` removes the cluster. Needs k3d and kubectl installed.

On AWS the same manifests run on EKS with a Fargate profile — scaffolded
in `aws/eks/create-cluster.yaml` and the `eks-up` / `eks-deploy` / `eks-decide` /
`eks-down` targets: the identical YAML, only the image reference swapped
for the ECR copy the aws/ stack built. The control plane bills hourly;
create, run, delete. Exposure stays at a port-forward — a production
ingress on EKS-Fargate needs the AWS Load Balancer Controller, which is
beyond this chapter. The chapter's first-class cloud path remains ECS.

Still to come in this chapter: the SageMaker endpoint the service can call
instead of its in-process scorecard, the Step Functions orchestration, the
`aws/` SAM template (Fargate service, desired-count 0 until you scale it),
and the burst load test.
