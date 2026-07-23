# Data Engineering on AWS: the bureau ELT pipeline

The running example for the data engineering chapter. Daily credit bureau
responses land as nested JSON in the raw zone of the lake; a Glue Python
shell job pulls one day's files, normalizes them with dlt, and loads them
to the warehouse. Transformations happen in the warehouse afterward (ELT).

The local cloud is real engines in Docker, no mocks:

- MinIO serving the S3 API (the raw zone and the feature store)
- redshift-local (Postgres 16 behind a Redshift wire proxy) as the warehouse:
  dlt's redshift destination and the dbt-redshift adapter both run against it
- an Iceberg REST catalog and Trino (the engine inside Athena), standing in
  for a SageMaker Feature Store offline store in the Iceberg table format
- DynamoDB Local serving the online-store pattern: latest feature row per
  applicant, fetched by key

## Run it

```
make up              # start MinIO, the warehouse, the catalog, Trino
make seed            # write yesterday's bureau files to the raw zone
make run             # the Glue job: read, normalize with dlt, load
make run-glue        # the same job inside the official Glue 5 runtime image
make query           # row counts and the evolved schema
make dbt-run         # staging views and the gold mart, via dbt-redshift
make dbt-test        # source and mart tests
make gold            # top of applicant_credit_profile
make lake-objects    # buckets, objects, prefixes, provenance metadata
make lake-parquet    # partitioned Parquet + a DuckDB query in place
make lake-table      # an Iceberg applications table via PyIceberg
make lake-table-query # the same table through Trino SQL
make features        # ship the gold mart to the Iceberg feature table
make features-query  # read it back through Trino
make features-lookup # one applicant's features from the online store
make down            # stop and clean
```

To see schema evolution, load a second day where the bureau added fields.
Start from a clean warehouse (`make down && make up`) so each applicant has
exactly one report per day loaded — `run` appends, so re-running the same
date, or overlapping with the quickstart's default (yesterday), stacks
duplicate reports for that applicant:

```
make down && make up
make seed DATE=2026-07-17 && make run DATE=2026-07-17
make seed-drift DATE=2026-07-18 && make run DATE=2026-07-18
make query       # employment__status and score__segment appeared; no run failed
```

Across multiple days an applicant has one report per day; the gold mart's
score is the latest report's, and its tradeline aggregates are scoped to
that same latest report.

## Local vs AWS

The job code never changes. Locally, `AWS_ENDPOINT_URL` points boto3 at
MinIO; the dlt destination is `redshift` in both worlds (redshift-local
accepts Redshift DDL such as `varchar(max)`), and dbt runs the real
`dbt-redshift` adapter. On AWS: unset the endpoint, point
`DESTINATION__REDSHIFT__CREDENTIALS` and `WAREHOUSE_HOST` at the real
warehouse, set `WAREHOUSE_SSLMODE` to a real mode, and deploy the job file
as a Glue Python shell job with `--additional-python-modules "dlt[redshift]"`.
The job parses arguments with `parse_known_args`, so Glue's own job
arguments pass through untouched.

The feature-store demo mirrors a SageMaker Feature Store offline store in
the Iceberg table format: locally the feature table lives in the REST
catalog over MinIO and Trino queries it; on AWS you create the feature
group with the Iceberg format and ship the same rows with
`feature_group.ingest()`. The online-store pattern runs on DynamoDB Local; only the featurestore-runtime API itself (GetRecord/PutRecord) has no local emulator.

## Layout

One local cloud for the chapter, one folder per use case:

- `docker-compose.yml`, `trino/`: the chapter's local cloud (MinIO,
  redshift-local, Iceberg REST catalog, Trino, DynamoDB Local)
- `bureau-elt/`: the daily bureau pipeline
  - `generate_bureau_files.py`: synthesizes a daily drop of nested bureau JSON
  - `glue_bureau_job.py`: the Glue job (list, read, normalize with dlt, load)
  - `dbt/`: sources with tests, staging views, and the gold mart
    `applicant_credit_profile`
- `lake-basics/`: the lake section's examples (objects and metadata,
  partitioned Parquet with DuckDB, an Iceberg table with PyIceberg).
  S3 Inventory configuration appears in the chapter text only: MinIO does
  not implement the inventory API, and the configuration has not yet been
  applied to a bucket on real S3
- `feature-store/`: the feature-store demo
  - `ingest_features.py`: gold mart to the Iceberg feature table and the
    online store
  - `lookup_features.py`: one applicant's features by key, the serving read
- `Makefile`: all targets above; `make lint` runs ruff
