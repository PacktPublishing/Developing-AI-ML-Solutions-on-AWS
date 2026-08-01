# Streaming fraud detection on AWS

The cloud round of the chapter: the same system the local stack runs, on the
real services. The SAM template creates the two Kinesis streams, the
`blocked_transactions` table, the ECR repository and staging bucket, the
Lambda consumer with its Kinesis event source, and the Firehose delivery
stream. The scoring endpoint deploys separately, because its image has to be
pushed first.

## Before the first deploy

The deploying user needs Kinesis, Firehose, ECR, and SageMaker grants that
a scoped account will not have by default; they are collected in
`iam/deploy.json`. Attach the policy once — this account keeps per-chapter
grants on IAM groups, because a user's inline and attached-policy budgets
run out:

```
aws iam create-group --group-name book-ch06
aws iam put-group-policy --group-name book-ch06 \
  --policy-name FraudBookCh6Deploy --policy-document file://iam/deploy.json
aws iam add-user-to-group --group-name book-ch06 --user-name <you>
```

## Run it, in order

```
make deploy          # the SAM stack
make image           # build and push the serving image to the stack's ECR repo
make endpoint-local  # rehearse: SageMaker local mode runs that image on :8080
make endpoint        # the real deploy: Serverless Inference from the same image
make produce         # replay 500 transactions at the real stream
make blocks          # count what the consumer blocked
make teardown        # endpoint, bucket contents, then the stack
```

`make endpoint-local` is SageMaker local mode, the chapter-02 pattern
applied to serving: the SDK keeps the real control plane — your role, the
image in ECR, the tarball path in S3 — but runs the container on this
machine and health-checks /ping. What serves on :8080 is byte-for-byte the
container the endpoint will run, and the streaming consumer's
invoke_endpoint call works against it unchanged.

`make produce` runs the chapter's own producer with `KINESIS_ENDPOINT`
cleared — the same code that replayed against kinesalite now talks to the
real stream, which is the whole point.

The consumer's `SCORE_THRESHOLD` parameter defaults to the value in
`artifacts/model_meta.json`; if you retrain, redeploy with the new threshold
(`sam deploy --parameter-overrides ScoreThreshold=<value>`).

## The Redshift leg

By default Firehose delivers staged batches to S3 only. To complete the
warehouse leg, deploy with the warehouse parameters:

```
sam deploy --parameter-overrides \
  RedshiftJdbcUrl=jdbc:redshift://<host>:5439/fraud \
  RedshiftUser=analyst RedshiftSecretArn=<secret-arn>
```

The secret holds the warehouse password under the key `password`; the
template reads it as a dynamic reference, so the credential never passes
through a stack parameter.

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
