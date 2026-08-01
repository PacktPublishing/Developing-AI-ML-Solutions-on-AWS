# Chapter code

The runnable code for *Developing AI/ML Solutions on AWS*, one folder per chapter.
Every chapter runs **locally first** — real engines in containers standing in for
the AWS services — then on **real AWS** with the same code; only the endpoints
change. Each chapter's `diagrams/` holds the cloud architecture and its local
mirror, and `make diagrams` regenerates them from `diagrams/yaml/`.

Below: what each chapter builds, the AWS services it uses, and the local stand-in
for each.

---

## Chapter 01 — Data engineering

The credit-bureau data lake and feature store: an ELT that lands raw files in S3,
loads a Redshift warehouse with dlt, builds dbt marts, and ships features to an
offline (Iceberg) and online (DynamoDB) store.

![Bureau ELT](ch-01-data-engineering/diagrams/png/fig_bureau_elt.png)

- **AWS services:** Amazon S3, S3 Tables, AWS Glue, Amazon Redshift Serverless,
  Amazon Athena, Amazon EventBridge, SageMaker Feature Store, Amazon DynamoDB,
  Airflow on EC2
- **Local stand-ins:** MinIO (S3), Iceberg REST catalog (Glue Data Catalog), Trino
  (Athena), redshift-local (Redshift), DynamoDB Local (Feature Store online),
  Glue + dlt

## Chapter 02 — MLOps

A champion/challenger scorecard trained on SageMaker, tracked in MLflow and tuned
with AMT, then served three ways — endpoint, Lambda, and Fargate — from one image.

![Serving surfaces](ch-02-mlops/diagrams/png/fig_serving_surfaces.png)

- **AWS services:** Amazon SageMaker AI (training, endpoints, Pipelines, MLflow,
  AMT), Amazon ECR, AWS CodeBuild, AWS Deep Learning Containers, AWS Lambda, AWS
  Fargate, Amazon API Gateway
- **Local stand-ins:** custom containers (docker run), serverless MLflow (sqlite +
  registry), S3Proxy (S3), Lambda RIE, Fargate via Docker Compose

## Chapter 03 — Generative AI

A grounded underwriting assistant on Bedrock: retrieval over pgvector, guardrails
on the answer path, and a Strands agent — the whole loop runnable offline.

![RAG pipeline](ch-03-generative-ai/diagrams/png/fig_rag_pipeline.png)

- **AWS services:** Amazon Bedrock (Converse, embeddings, Guardrails, AgentCore),
  Amazon RDS for PostgreSQL + pgvector, AWS Lambda, Amazon API Gateway, AWS Secrets
  Manager
- **Local stand-ins:** Ollama (Bedrock), pgvector Postgres (RDS), Strands agent
  (`BEDROCK_LOCAL=1`)

## Chapter 04 — Realtime scoring

The decision service: a scoring container behind an Application Load Balancer on
ECS Fargate, logging every decision to DynamoDB. The image is built in the cloud
by CodeBuild — the laptop never builds it.

![Decision service](ch-04-realtime-scoring/diagrams/png/fig_decision_service.png)

- **AWS services:** Amazon ECS on AWS Fargate, Application Load Balancer, Amazon
  DynamoDB, AWS CodeBuild, Amazon ECR, Amazon S3, Amazon VPC
- **Local stand-ins:** the service container + DynamoDB Local (Docker Compose)

## Chapter 05 — Batch limits

The monthly batch run: an EventBridge schedule starts a SageMaker pipeline that
recomputes credit limits against the Redshift warehouse, stages results in S3, and
alerts an SNS topic on failure.

![Batch pipeline](ch-05-batch-limits/diagrams/png/fig_batch_pipeline.png)

- **AWS services:** Amazon EventBridge (schedule), SageMaker Pipelines, AWS
  CodeBuild, Amazon ECR, Amazon S3, Amazon Redshift (warehouse from ch-01), Amazon
  SNS
- **Local stand-ins:** redshift-local (Redshift), S3Proxy (S3), step container
  (Docker Compose)

---

## Per-chapter deploy identities

Each chapter deploys with its **own** scoped IAM role — `ch01-user` … `ch08-user` —
carrying only that chapter's permissions, instead of running everything as admin.
Piling every chapter's permissions onto one admin user hits the IAM per-user policy
quota (the "runs out of permissions" failure); one assumable role per chapter avoids
that and keeps each chapter least-privilege. Each role's permissions live in that
chapter's `aws/iam/deploy.json`.

Bootstrap once, from a privileged identity that can create roles:

```bash
code/setup-users.sh create arn:aws:iam::<ACCOUNT_ID>:user/<your-base-user>
```

That creates all eight roles, each trusting `<your-base-user>` to `sts:AssumeRole`
and carrying its chapter's `deploy.json` inline. Re-run it after editing any
`deploy.json` (updates in place); `code/setup-users.sh delete` removes them all.

Add a profile per chapter to `~/.aws/config` and deploy under it:

```ini
[profile ch04]
role_arn = arn:aws:iam::<ACCOUNT_ID>:role/ch04-user
source_profile = default
region = us-east-1
```

```bash
make -C code/ch-04-realtime-scoring deploy AWS_PROFILE=ch04
```

**These policies are best-effort and must be author-tested.** Each `deploy.json`
was derived from the chapter's `aws/` template and Make targets, not from a live
deploy. Least-privilege is iterative: assume the role, deploy, and if a call returns
`AccessDenied`, add that one action to the chapter's `deploy.json` and re-run the
bootstrap. A few statements stay broad (`ec2:*Vpc*`, `glue:*`, `sagemaker:*`) where
tight ARN scoping isn't practical.
