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
make airflow-deploy  # Airflow on EC2 (SAM, airflow-ec2/); free-tier redshift-local warehouse
                     #   WAREHOUSE=serverless SUBNETS=subnet-a,subnet-b,subnet-c -> Redshift Serverless
make airflow-delete
```

## Observed Feature Store semantics (verify against docs before print)

Parity matrix identical local vs AWS except deletes: after DeleteRecord
(SoftDelete, tombstone EventTime strictly newer), real BatchGetRecord
excludes the record immediately, but real GetRecord kept returning it for
60+ seconds. The local shim deletes synchronously from both read paths.

## The warehouse: free tier or serverless

Airflow is the chapter's orchestrator; its DAG loads a warehouse. The one SAM
stack in `airflow-ec2/` provisions both options — pick the mode at deploy:

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
