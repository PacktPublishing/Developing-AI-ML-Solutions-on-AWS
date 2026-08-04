# Streaming fraud detection on AWS

The cloud round of the chapter: the same system the local stack runs, on the
real services. One SAM template creates the two Kinesis streams, the
`blocked_transactions` table, the ECR repository and staging bucket, the
Lambda consumer with its Kinesis event source, the Firehose delivery stream,
and — with `EnableWarehouse=true`, the default — the Redshift Serverless
warehouse Firehose COPYs decisions into. The scoring endpoint deploys
separately, because its image has to be pushed first.

## Before the first deploy

`make iam` provisions this chapter's deploy identity: it creates `ch06-user`
from `iam/deploy.json` (least-privilege for ch-06 only) and trusts you to
assume it. Then add a `[profile ch06]` (`role_arn` + `source_profile`) and run
every cloud target with `AWS_PROFILE=ch06`, so the whole round runs under the
same scoped grants a reader gets:

```
make iam        # create/refresh ch06-user from iam/deploy.json
# add [profile ch06] to ~/.aws/config, then: export AWS_PROFILE=ch06
```

## Run it, in order

```
make deploy          # the SAM stack, warehouse included
make seed-warehouse  # create fraud_decisions in Redshift (Data API)
make image           # build and push the serving image to the stack's ECR repo
make endpoint-local  # rehearse: SageMaker local mode runs that image on :8080
make endpoint        # the real deploy: Serverless Inference from the same image
make produce         # replay 500 transactions at the real stream
make blocks          # count what the consumer blocked
make query           # the analyst's view: blocks and passes in the warehouse
make teardown        # endpoint, bucket contents, then the stack — warehouse and all
```

`make endpoint-local` is SageMaker local mode, the chapter-02 pattern
applied to serving: the SDK keeps the real control plane — your role, the
image in ECR, the tarball path in S3 — but runs the container on this
machine. Its readiness check invokes the endpoint with the SchemaBuilder's
sample record (not a bare /ping), so `deploy.py` gives it one fully-featured
transaction. What serves on :8080 is byte-for-byte the container the endpoint
will run, and the streaming consumer's invoke_endpoint call works against it
unchanged.

`make produce` runs the chapter's own producer with `KINESIS_ENDPOINT`
cleared — the same code that replayed against kinesalite now talks to the
real stream, which is the whole point.

The consumer's `SCORE_THRESHOLD` parameter defaults to the value in
`artifacts/model_meta.json`; if you retrain, redeploy with the new threshold
(`sam deploy --parameter-overrides ScoreThreshold=<value>`).

## The Redshift leg

The warehouse is part of the same stack. `EnableWarehouse` (default `true`)
builds a publicly accessible Redshift Serverless workgroup, its security group
open only to the Amazon Data Firehose service CIDR on 5439, plus a generated
admin secret; the Firehose delivery stream then targets Redshift instead of
S3. Set `EnableWarehouse=false` to skip it and have Firehose stage to S3 only
(run the COPY from the staged objects later). No second stack, no external JDBC
parameter — the delivery stream reads the workgroup endpoint and the secret
straight from the template.

`make seed-warehouse` creates the `fraud_decisions` table through the Redshift
Data API (no VPC path needed), authenticating with that generated secret; the
secret holds the password under the key `password`, which the Firehose config
reads as a dynamic reference so the credential never sits in a parameter.

Firehose then issues `COPY fraud_decisions FROM <staged object>` — the same
choreography the local/firehose-local shim performs, from the same destination
configuration `src/streaming/downstream.py` passes to the API locally. One
divergence to know: real Redshift COPYs the staged JSON directly (`FORMAT AS
JSON 'auto'`); the local warehouse is Postgres, whose COPY has no JSON mode,
so the shim reshapes each staged batch to CSV before loading it.

## Costs

Everything here is either pay-per-request or has a small hourly price: two
provisioned shards, the serverless endpoint (scale-to-zero), Firehose per GB.
Nothing should run overnight — `make teardown` deletes the endpoint, empties
the staging bucket, and removes the stack; every stateful resource carries a
Delete policy and the ECR repository empties itself on delete.
