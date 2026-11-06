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

![The serving layer](ch-02-mlops/diagrams/png/fig_serving_layer.png)

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

## Chapter 06 — Streaming fraud

Real-time fraud scoring on a transaction stream: records land on Kinesis, a
scoring function flags them, results are archived through Firehose and logged to
DynamoDB, with the credit warehouse alongside for enrichment.

![Streaming fraud](ch-06-streaming-fraud/diagrams/png/fig_streaming.png)

- **AWS services:** Amazon Kinesis Data Streams, Amazon Data Firehose, AWS Lambda,
  Amazon DynamoDB, Amazon Redshift Serverless, Amazon S3, AWS CodeBuild, Amazon ECR
- **Local stand-ins:** kinesis-local (Kinesis), a from-source Firehose, DynamoDB
  Local, S3Proxy (S3), redshift-local (Redshift)

## Chapter 07 — Vision RAG

KYC face verification as vector search: a face embedder turns an ID and a selfie
into vectors, pgvector finds the nearest registered identity, and the match is
verified against the claimed subject — the whole loop behind an HTTP API.

![Serverless KYC](ch-07-vision-rag/diagrams/png/fig_serverless_kyc.png)

- **AWS services:** Amazon RDS for PostgreSQL + pgvector, AWS Lambda, Amazon API
  Gateway (HTTP API), Amazon S3, AWS CodeBuild, Amazon ECR
- **Local stand-ins:** pgvector Postgres (RDS); the face embedder/explainer run
  directly

## Chapter 08 — Self-service analytics

Analysts ask questions in plain English; Claude Code on Bedrock writes read-only
SQL, runs it through the Redshift Data API under a scoped task role, and answers
with a table, a summary, and the tables it touched — the terminal in the browser.

![The analytics assistant](ch-08-selfservice-analytics/diagrams/png/fig_assistant.png)

- **AWS services:** Amazon Bedrock, Amazon Redshift Serverless (Data API), Amazon
  ECS Fargate, Amazon ECR, AWS CodeBuild, Amazon CloudWatch Logs
- **Local stand-ins:** redshift-local + a from-source Data API shim (Redshift),
  Docker Compose (Fargate), Ollama / vllm-metal (Bedrock)

## Chapter 09 — Underwriting knowledge base

Policy documents, credit reports, and synthetic dossier memos embedded with
Bedrock and stored in OpenSearch, so a question is answered with citations to the
source loan or policy id and a Strands agent can assemble similar past cases into
a recommendation the underwriter signs off on. The underwriter interface is a
FastAPI ask/recommend app that runs under uvicorn locally and, unchanged, in a
container Lambda behind API Gateway.

![The retrieval round on AWS](ch-09-underwriting-kb/diagrams/png/fig_aws.png)

- **AWS services:** Amazon Bedrock (embeddings, Converse), Amazon OpenSearch
  Service (domain and Dashboards), Amazon Cognito, AWS Lambda (container image),
  Amazon API Gateway (HTTP API), Amazon CloudWatch Logs
- **Local stand-ins:** Ollama (Bedrock), an OpenSearch container with Dashboards
  (OpenSearch Service), a from-source user pool (Cognito), uvicorn (Lambda and
  API Gateway)

## Chapter 10 — Conversation classification

The pile every support and collections team knows: thousands of conversations
nobody has time to skim. Each one is classified into a category with zero-shot
inference and handed to the right team through SNS, with a labeled set keeping the
accuracy honest. The work runs as an Amazon Bedrock batch inference job: every
conversation is one line of a JSONL manifest in S3, and Bedrock replays each
`modelInput` against the model and writes the answers back.

![The classification round on AWS](ch-10-conversation-classification/diagrams/png/fig_batch.png)

- **AWS services:** Amazon Bedrock (batch inference, Converse), Amazon SNS, Amazon S3
- **Local stand-ins:** Ollama (Bedrock), a from-source batch-inference runner
  (Bedrock batch), a from-source SNS shim delivering to a file inbox per team
  (SNS), S3Proxy (S3)

## Chapter 11 — Monitoring

Online Population Stability Index monitoring of the scores and the features that
drive them, binned at the model's own CatBoost split borders, SHAP attribution
drift reported as the NDCG of the current feature ranking against the reference
one, and CloudWatch alerting on top. SageMaker Model Monitor and Clarify are
closed to new accounts, so the AWS round is AWS's documented replacement: a
SageMaker Pipeline running Batch Transform into a Processing job that writes the
Clarify-shaped artifacts and publishes the metrics the alarms watch.

![The monitoring pipeline](ch-11-model-monitoring/diagrams/png/fig_pipeline.png)

- **AWS services:** Amazon SageMaker AI (Pipelines, Batch Transform, Processing),
  Amazon ECR, AWS CodeBuild, Amazon S3, Amazon CloudWatch (metrics and alarms)
- **Local stand-ins:** SageMaker local mode for the training and monitoring
  containers, S3Proxy (S3), a from-source CloudWatch shim (CloudWatch); SHAP and
  CatBoost run directly

## Chapter 12 — Governance

Model cards, the model registry, and lineage for auditors; least-privilege IAM;
and a new model shipped with champion/challenger rollout, flipped through
AppConfig feature flags with no redeploy. A FastAPI gateway routes each
application to chapter 2's WOE scorecard or its monotone CatBoost challenger by
the `challenger_rollout` flag, whose `split` rule buckets each loan
deterministically, so widening the rollout is a flag edit and no loan flips back.

![The rollout and governance stack](ch-12-model-governance/diagrams/png/fig_aws.png)

- **AWS services:** AWS AppConfig (with the AppConfig agent as an ECS sidecar),
  Amazon ECS Fargate, Amazon ECR, AWS CodeBuild, Amazon SageMaker AI (endpoints,
  Model Cards, Model Registry, ML Lineage), AWS IAM
- **Local stand-ins:** the published AppConfig agent image in local development
  mode, reading an Ion feature-flag file (AppConfig); Docker Compose (Fargate);
  chapter 2's serving containers (SageMaker endpoints); the cards, registry, and
  lineage written as JSON (SageMaker governance)

---

## Per-chapter deploy identities

Each chapter deploys with its **own** scoped IAM role — `ch01-user` … `ch12-user` —
carrying only that chapter's permissions, instead of running everything as admin.
Piling every chapter's permissions onto one admin user hits the IAM per-user policy
quota (the "runs out of permissions" failure); one assumable role per chapter avoids
that and keeps each chapter least-privilege. Each role's permissions live in that
chapter's `aws/iam/deploy.json`.

**Provision it per chapter, when you reach that chapter's cloud work — not the
whole book up front.** Each chapter's `aws/Makefile` has an `iam` target; from a
privileged identity (one that can create roles):

```bash
make -C code/ch-04-realtime-scoring/aws iam    # creates ch04-user from its deploy.json
```

`make iam` creates `chNN-user` trusting you to `sts:AssumeRole` and attaches that
chapter's `deploy.json` inline. Re-run it after editing the policy.

Then add a profile for the chapter to `~/.aws/config` and deploy under it:

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
was derived from the chapter's `aws/` and Make targets. Least-privilege is
iterative: run the chapter as `chNN-user`, and if a call returns `AccessDenied`,
add that one action to the chapter's `deploy.json` and re-run `make iam`. A few
statements stay broad (`ec2:*Vpc*`, `glue:*`, `sagemaker:*`) where tight ARN
scoping isn't practical.
