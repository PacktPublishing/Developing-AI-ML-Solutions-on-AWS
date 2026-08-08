# Cloud verification

Everything in this chapter runs locally first. This folder is the proof
that the same code runs on real AWS: each target executes the identical
script with the endpoint unset, so boto3 falls through to the default
credential chain. Each target tees its output to `/tmp` with the date.

```
make check-account   # which account and bucket names the runs will use
make lake-objects    # buckets/objects/metadata on real S3
make lake-parquet    # partitioned Parquet to real S3, awswrangler reads it back
make lake-table      # S3 Tables: PyIceberg append + scan over SigV4 REST
make glue-run        # the bureau job as a real Glue Python shell job
make feature-group   # Feature Store online: create, put, get
make clean-feature-group
make clean-lake      # remove the verification lake bucket
make airflow-deploy  # Airflow on EC2 (SAM); free-tier redshift-local warehouse
                     #   WAREHOUSE=serverless SUBNETS=subnet-a,subnet-b,subnet-c -> Redshift Serverless
make airflow-password  # fetch the generated UI password (user: admin) over SSM
make airflow-delete
```

The Airflow UI is at the stack's `AirflowUrl` output, on port 8080 (open only to
your IP). Airflow 3 standalone generates the `admin` password on the instance;
`make airflow-password` reads it back over SSM — no SSH and no extra open port.

### Trigger the DAG without the UI (REST API)

Airflow 3 serves a REST API on the same `:8080`. Get a bearer token with the
admin password, then unpause and trigger the DAG — the same calls a scheduler
or CI would make (no SSH into the box):

```bash
URL=http://<AirflowUrl>        # the stack output
PW=<make airflow-password>     # the admin password
TOKEN=$(curl -s -X POST "$URL/auth/token" -H 'content-type: application/json' \
  -d "{\"username\":\"admin\",\"password\":\"$PW\"}" | jq -r .access_token)

curl -s -X PATCH "$URL/api/v2/dags/bureau_pipeline?update_mask=is_paused" \
  -H "Authorization: Bearer $TOKEN" -H 'content-type: application/json' \
  -d '{"is_paused": false}'
curl -s -X POST "$URL/api/v2/dags/bureau_pipeline/dagRuns" \
  -H "Authorization: Bearer $TOKEN" -H 'content-type: application/json' \
  -d '{"logical_date": null}'
```

Poll the run's `state` at `GET /api/v2/dags/bureau_pipeline/dagRuns/{run_id}` until
it reads `success`.

## Observed Feature Store semantics (verify against docs before print)

Parity matrix identical local vs AWS except deletes: after DeleteRecord
(SoftDelete, tombstone EventTime strictly newer), real BatchGetRecord
excludes the record immediately, but real GetRecord kept returning it for
60+ seconds. The local shim deletes synchronously from both read paths.

## The warehouse: free tier or serverless

Airflow is the chapter's orchestrator; its DAG loads a warehouse. The one SAM
stack (`template.yaml`) provisions both options, pick the mode at deploy:

- **`WAREHOUSE=local` (default, free tier):** redshift-local runs on the
  instance under docker compose, mirroring the local stack. Nothing else to set.
- **`WAREHOUSE=serverless` (standard plan):** the same stack also provisions
  Amazon Redshift Serverless (namespace + 8-RPU workgroup, private, reachable
  only from the instance), generates the admin password into Secrets Manager,
  and points the DAG at it — the DAG reads the secret at run time with the
  instance role, so nothing is typed or baked into UserData. Pass the workgroup
  subnets (>= 3 across AZs):

  ```
  make airflow-deploy WAREHOUSE=serverless SUBNETS=subnet-a,subnet-b,subnet-c
  ```

Redshift Serverless bills per second while active; `make airflow-delete` tears
the whole stack (warehouse included) down. A Free Plan account blocks Redshift
Serverless — use the default local mode there.

## One-time Glue setup (already applied in this account)

- role `ch01-glue`: trusts glue.amazonaws.com, AWSGlueServiceRole plus
  S3 access scoped to the raw bucket
- job `bureau-elt-ch01`: pythonshell, Python 3.9, 0.0625 DPU, script at
  `s3://bureau-raw-<account>/scripts/glue_bureau_job.py`
- re-upload the script after changes:
  `aws s3 cp src/bureau-elt/glue_bureau_job.py s3://bureau-raw-<account>/scripts/`
