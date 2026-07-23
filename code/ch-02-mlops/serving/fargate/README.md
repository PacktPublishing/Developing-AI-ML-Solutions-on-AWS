# Fargate serving

The challenger serving image as a long-lived container: it already serves
`/ping` and `/invocations` on 8080 (the SageMaker inference contract), so ECS
Fargate runs it unchanged behind an ALB or API Gateway.

- `Dockerfile`: wraps the challenger image; `make fargate-build` bakes the
  model in, so the task is self-contained
- `docker-compose.yml`: the local task runner (`make fargate-local`, port 8093)
- `docker-compose.override.yml`: emulates the ECS task-metadata and
  task-role credential endpoints via `amazon-ecs-local-container-endpoints`
- `model/`: the baked-in artifact, copied by `make fargate-build` (git-ignored)

On AWS the same image runs as a Fargate task with the `ch02-ecs-exec-role`
execution role (`aws/README.md` covers the IAM). A Fargate service bills while
it runs, so stop tasks when done.
