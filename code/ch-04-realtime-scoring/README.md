# Real-time credit scoring behind a gateway

The running example for the real-time chapter. A lender's core product moment
is a loan application answered while the applicant waits: the service takes the
application, the models give a probability of default, and the policy turns that
— together with hard rules — into approve, refer, or decline, with the reason
codes an adverse-action notice is built from. The decision, not the score, is the
product.

The service is a FastAPI **gateway** with typed contracts. It does not score
in-process; it orchestrates several model endpoints. It fans the application out
to two SageMaker endpoints — an incumbent scorecard and a challenger — averages
their probabilities into one ensemble PD, applies the KYC and policy rules, and
logs the decision, with each model's score, to DynamoDB. The gateway knows nothing
about where the models run: each entry in `ENDPOINTS` is a name and a target, a URL
locally or a SageMaker endpoint name in the cloud.

## Run it (offline, no AWS)

Docker Compose runs both model containers and the gateway; the gateway reaches
each model on the compose network and writes decisions to DynamoDB Local.

```
make up        # both model endpoints, the gateway, and the decision log
make decide    # a clean application: APPROVE
make refer     # the gray band: REFER, with the reason
make decline   # failed KYC: DECLINE, whatever the score says
make log       # the audit trail in DynamoDB
make down      # stop and clean
```

Each model is a bring-your-own serving container — the SageMaker inference
contract, `GET /ping` and `POST /invocations` on 8080 — so the same image runs
here under Docker and on a SageMaker endpoint in the cloud.

## Two ways to run the models locally

- **Docker Compose (above).** Fully offline, no AWS account: the containers run
  as compose services and the gateway invokes them over HTTP.
- **SageMaker local mode.** `make sm-local MODEL=scorecard SM_ROLE=<role>` deploys
  a model with the SDK's `ModelBuilder` in `Mode.LOCAL_CONTAINER` — the same
  deploy code the cloud uses, with the mode as the only difference. This runs the
  container through the SageMaker SDK rather than plain Docker, so it needs real
  AWS credentials, a SageMaker execution role, and the image in ECR
  (`make -C aws push-models`); it creates no cloud resources. `make -C aws iam`
  prints the role ARN to pass as `SM_ROLE`.

## Local vs AWS

| In this directory | On AWS |
| --- | --- |
| model containers on the compose network | SageMaker endpoints (scorecard, challenger) |
| the gateway container on :8080 | the same image on ECS Fargate behind an ALB |
| `ENDPOINTS=name=http://…` | `ENDPOINTS=name=<endpoint>` |
| DynamoDB Local decision log | DynamoDB |

## The whole thing on Kubernetes (local EKS)

`make kube-up` starts a k3d cluster (k3s in Docker), imports the three images,
and applies `k8s/ensemble.yaml` — the gateway and both models as Deployments and
Services, the gateway reaching each model through a cluster Service, with `/health`
wired to the readiness and liveness probes and an Ingress. `make kube-decide`
answers through it; `make kube-down` removes the cluster. Needs k3d and kubectl.

This is the local stand-in for EKS. On AWS the models are SageMaker endpoints
rather than pods, so the EKS path runs the **gateway** alone (`k8s/gateway.yaml`)
and it invokes the endpoints — `make -C aws deploy COMPUTE=eks`.

The `aws/` directory has the SAM stack, the SageMaker endpoint deploys, and the
IAM: see [`aws/README.md`](aws/README.md).
