# Cloud verification

Everything in this chapter runs locally first. This folder is the proof
that the same code runs on real AWS: each target executes the identical
script with the endpoint unset, so boto3 falls through to the default
credential chain. Each target tees its output to `/tmp` with the date.

```
make check-account   # which account and bucket names the runs will use
make lake-objects    # buckets/objects/metadata on real S3
make lake-parquet    # partitioned Parquet to real S3, DuckDB reads it back
make lake-table      # S3 Tables: PyIceberg append + scan over SigV4 REST
make glue-run        # the bureau job as a real Glue Python shell job
make feature-group   # Feature Store online: create, put, get
make clean-feature-group
make clean-lake      # remove the verification lake bucket
make airflow-deploy  # single-node Airflow on EC2 (SAM, airflow-ec2/)
make airflow-delete
```

## Observed Feature Store semantics (verify against docs before print)

Parity matrix identical local vs AWS except deletes: after DeleteRecord
(SoftDelete, tombstone EventTime strictly newer), real BatchGetRecord
excludes the record immediately, but real GetRecord kept returning it for
60+ seconds. The local shim deletes synchronously from both read paths.

## Redshift Serverless

A Free Plan account blocks Redshift Serverless; a standard plan runs it.
Provision with `redshift-serverless.yaml` (CloudFormation): namespace +
workgroup (8 RPU), publicly accessible, SG for 5439 from one IP,
`require_ssl`. The bureau job loads it through dlt (explicit
`DESTINATION__REDSHIFT__CREDENTIALS__*`) and dbt runs and tests against it
with verify-full SSL on the default Amazon CA.

```
aws cloudformation deploy --template-file redshift-serverless.yaml \
  --stack-name ch01-redshift --region us-east-1 \
  --parameter-overrides VpcId=... SubnetIds=...,...,... AllowedCidr=x.x.x.x/32 AdminPassword=...
```

Billing is per-second while the workgroup is active, so tear down the same
day: `aws cloudformation delete-stack --stack-name ch01-redshift --region us-east-1`.

## One-time Glue setup (already applied in this account)

- role `glue-book-ch01`: trusts glue.amazonaws.com, AWSGlueServiceRole plus
  S3 access scoped to the raw bucket
- job `bureau-elt-ch01`: pythonshell, Python 3.9, 0.0625 DPU, script at
  `s3://bureau-raw-<account>/scripts/glue_bureau_job.py`
- re-upload the script after changes:
  `aws s3 cp bureau-elt/glue_bureau_job.py s3://bureau-raw-<account>/scripts/`
