# The same challenger serving image, run as a container on ECS Fargate. It
# already serves /ping and /invocations on 8080 (the SageMaker inference
# contract), so Fargate just runs it as a long-lived task behind API Gateway or
# an ALB — no code changes from the endpoint. The model is baked in by
# `make fargate-build` (git-ignored), so the task is self-contained.
#
# BASE_IMAGE defaults to the ECR image (what CodeBuild pulls); for a local build
# pass --build-arg BASE_IMAGE=ch02-challenger:latest.
ARG BASE_IMAGE=123456789012.dkr.ecr.us-east-1.amazonaws.com/ch02-challenger:latest
FROM ${BASE_IMAGE}

COPY model/ /opt/ml/model/
EXPOSE 8080
CMD ["serve"]
