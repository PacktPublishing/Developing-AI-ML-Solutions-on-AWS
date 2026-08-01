# TOC

Every chapter runs locally first — real engines in containers standing in for the
AWS services — then on real AWS with the same code; only the endpoints change.
Below is what each chapter uses on AWS and the local stand-in for each.

- **Chapter 01 — Data engineering**
  - AWS Services covered: Amazon S3, S3 Tables, AWS Glue, Amazon Redshift
    Serverless, Amazon Athena, Amazon EventBridge, SageMaker Feature Store,
    Amazon DynamoDB, Airflow on EC2
  - Local AWS Services covered: MinIO (S3), Iceberg REST catalog (Glue Data
    Catalog), Trino (Athena), redshift-local (Redshift), DynamoDB Local (Feature
    Store online), Glue + dlt
- **Chapter 02 — MLOps**
  - AWS Services covered: Amazon SageMaker AI (training, endpoints, Pipelines,
    MLflow, AMT), Amazon ECR, AWS CodeBuild, AWS Deep Learning Containers, AWS
    Lambda, AWS Fargate, Amazon API Gateway
  - Local AWS Services covered: custom containers (docker run), serverless MLflow
    (sqlite + registry), S3Proxy (S3), Lambda RIE, Fargate via Docker Compose
- **Chapter 03 — Generative AI**
  - AWS Services covered: Amazon Bedrock (Converse, embeddings, Guardrails,
    AgentCore), Amazon RDS for PostgreSQL + pgvector, AWS Lambda, Amazon API
    Gateway, AWS Secrets Manager
  - Local AWS Services covered: Ollama (Bedrock), pgvector Postgres (RDS),
    Strands agent (BEDROCK_LOCAL=1)
- **Chapter 04 — Realtime scoring**
  - AWS Services covered: Amazon ECS on AWS Fargate, Application Load Balancer,
    Amazon DynamoDB, AWS CodeBuild, Amazon ECR, Amazon S3, Amazon VPC
  - Local AWS Services covered: the service container + DynamoDB Local (Docker
    Compose)
- **Chapter 05 — Batch limits**
  - AWS Services covered: Amazon EventBridge (schedule), SageMaker Pipelines, AWS
    CodeBuild, Amazon ECR, Amazon S3, Amazon Redshift (warehouse from ch-01),
    Amazon SNS
  - Local AWS Services covered: redshift-local (Redshift), S3Proxy (S3), step
    container (Docker Compose)
