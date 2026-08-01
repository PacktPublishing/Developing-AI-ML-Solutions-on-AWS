# Streaming fraud detection

The running example for the streaming chapter. A fintech's card processor
asks for a decision while the terminal still shows "Processing Payment":
approve or decline, before the cardholder notices a delay. Behind that
decision sits a CatBoost classifier trained on three families of features —
the raw transaction, the transaction against the user's own history, and the
transaction against the merchant's history — the same families Revolut
describes for its card fraud detection system.

The chapter builds the system in four moves:

- a producer replays transactions onto a Kinesis stream, partitioned by user
- a consumer scores each event and writes a block to DynamoDB when the model
  flags fraud — the processor checks for a block by transaction id, and a
  miss means the payment goes through
- every decision, blocked or passed, goes onward to a second stream for
  analytics
- a local Firehose — a small shim the book owns — delivers every decision
  the way the real service does: buffer, stage to S3, COPY into Redshift

The local cloud is real engines, no mocks: the Kinesis API served by
kinesalite, the official DynamoDB Local, S3Proxy for the staging bucket, and
a Redshift-flavored Postgres for the warehouse. boto3 talks to all of them
through `endpoint_url`; point the same code at AWS by unsetting four
environment variables.

## Run it

```
make up        # start the local cloud
make seed      # synthesize the transaction stream and the training split
make train     # fit the classifier, freeze the operating threshold
make serve     # the model behind the SageMaker serving contract, port 8080
make produce   # replay 2,000 transactions onto the stream
make score     # drain the stream: blocks to DynamoDB, decisions onward
make firehose  # the local Firehose service, in its own terminal
make deliver   # create the delivery stream (same config as the template)
make query     # the analyst's view: blocks and passes in the warehouse
make lookup TX=tx-0000085    # the fast-path check: block found, or a miss
make down      # stop the stack and the serving container
make clean     # down, plus remove generated data, artifacts, and the image
```

`produce` and `score` can run in two terminals at once — the consumer exits
on its own once the stream has been quiet for a few seconds. The firehose
service keeps delivering for as long as it runs.

## Local vs AWS

| In this directory | On AWS |
| --- | --- |
| kinesalite container | Kinesis Data Streams |
| `streaming/consumer.py` polling shards | Lambda with a Kinesis event source |
| `make serve`: the scoring image on port 8080 | the same image behind SageMaker Serverless Inference |
| DynamoDB Local | DynamoDB |
| the firehose-local shim | Amazon Data Firehose with a Redshift destination |
| S3Proxy staging bucket | the Firehose staging bucket in S3 |
| redshift-local container | Amazon Redshift |

The consumer never loads the model: it calls `invoke_endpoint` through
boto3's sagemaker-runtime client in both worlds. The serving container
accepts the `/endpoints/<name>/invocations` path that client sends, so
locally the only difference is the endpoint URL environment variable.

The cloud deploy is `aws/deploy_serverless.py`: the same image from ECR and
the same model.tar.gz behind a real Serverless Inference endpoint, following
the chapter-02 BYOC pattern.

The shim mirrors Firehose deliberately: `streaming/downstream.py` calls
`create_delivery_stream` with the same `RedshiftDestinationConfiguration`
that `aws/template.yaml` declares — one destination config, two worlds — and
the delivery loop keeps the hour-partitioned key layout
(`decisions/YYYY/MM/DD/HH/...`), the buffer-by-size-or-time behavior, and a
real `COPY`. Firehose never inserts rows one by one, it stages a batch and
has the warehouse load it.

## The model

CatBoost with balanced class weights — the model this system deploys to
SageMaker Serverless Inference, and the one Revolut's card-fraud team chose
for the same reasons: gradient boosting on trees is robust on heterogeneous
features, needs little tuning, and scores a single row inside a real-time
budget. Fraud is rare in the training data (about half a percent), the split
is chronological so the model is judged on the future, and the operating
threshold is frozen at training time at a target precision of 30% — roughly
Revolut's published operating point, three false alarms per fraud caught,
because a blocked card is a recoverable annoyance and a fraudulent charge is
not. The artifact is `artifacts/model.cbm`, CatBoost's native format — the
same file a SageMaker model tarball carries to the serverless endpoint.
