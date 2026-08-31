"""The full nightly pipeline orchestrated by Airflow, end to end.

Stage the raw bureau files, load them with dlt, build the dbt marts, test them, then
ingest the result into the feature store. The same file runs on the laptop and on the
EC2 deployment: every target is read from the environment, so the local stack falls
back to the defaults below and the deployed stack overrides them with Redshift
Serverless, real S3, and a SageMaker feature group.
"""

import os
from datetime import datetime

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG

ENV = {
    "PATH": "/home/airflow/.local/bin:/usr/local/bin:/usr/bin:/bin",
    "HOME": "/tmp",
    "UV_CACHE_DIR": "/tmp/uvcache",
    "AWS_ENDPOINT_URL": "http://s3:9000",
    "AWS_ACCESS_KEY_ID": "local",
    "AWS_SECRET_ACCESS_KEY": "localsecret",
    "AWS_DEFAULT_REGION": "us-east-1",
    "WAREHOUSE_DSN": "postgresql://loader:loader@warehouse:5439/bureau",
    "WAREHOUSE_HOST": "warehouse",
    "WAREHOUSE_USER": "loader",
    "WAREHOUSE_PASSWORD": "loader",
    "WAREHOUSE_SSLMODE": "verify-ca",
    "SSL_CERT_FILE": "/tmp/ca-bundle.pem",
    # dlt's redshift destination reads these; in-container the warehouse is a
    # service name, not localhost
    "DESTINATION__REDSHIFT__CREDENTIALS__HOST": "warehouse",
    "DESTINATION__REDSHIFT__CREDENTIALS__PORT": "5439",
    "DESTINATION__REDSHIFT__CREDENTIALS__USERNAME": "loader",
    "DESTINATION__REDSHIFT__CREDENTIALS__PASSWORD": "loader",
    "DESTINATION__REDSHIFT__CREDENTIALS__DATABASE": "bureau",
    "ICEBERG_REST_URI": "http://catalog:8181",
    "ONLINE_STORE_URL": "http://online-store:8000",
    "DBT_PROFILES_DIR": "/work/src/bureau-elt/dbt",
    "DATE": "2026-08-16",
}
# the deployed stack sets these in the container; anything it sets wins, so one DAG
# file serves both without a mode flag
ENV.update({k: os.environ[k] for k in list(ENV) if os.environ.get(k)})
# S3Proxy's endpoint and its dummy credentials are local-only. On the deployed
# stack none of them is set, so they come out and the SDK falls back to the
# instance role against real S3. The trust bundle is built per run either way.
if not os.environ.get("AWS_ENDPOINT_URL"):
    for key in ("AWS_ENDPOINT_URL", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        ENV.pop(key, None)
DBT = "uvx --from dbt-core==1.12.2 --with dbt-redshift==1.11.0 dbt"
# redshift-local mints a fresh certificate every time it starts, so the trust
# bundle is rebuilt before dbt connects rather than kept on disk: certifi's roots,
# plus the warehouse's own certificate when the warehouse is that container. Real
# Redshift chains to a public root and needs only the first half. This is the
# certs target from the chapter Makefile, which dbt-run and dbt-test depend on.
CERTS = (
    "uv run --with certifi python -c "
    "\"import certifi, shutil; shutil.copy(certifi.where(), '/tmp/ca-bundle.pem')\" && "
    'if [ "$WAREHOUSE_HOST" = warehouse ]; then '
    "openssl s_client -starttls postgres -connect warehouse:5439 </dev/null 2>/dev/null "
    "| openssl x509 -outform pem >> /tmp/ca-bundle.pem; fi; "
)
W = "set -e; cd /work; "

with DAG(
    "bureau_full_pipeline",
    start_date=datetime(2026, 1, 1),
    # unscheduled on the laptop, where you trigger it by hand; the deployed stack
    # sets DAG_SCHEDULE to @daily so it runs nightly on its own
    schedule=os.environ.get("DAG_SCHEDULE") or None,
    catchup=False,
) as dag:
    seed = BashOperator(
        task_id="seed",
        append_env=True,
        env=ENV,
        bash_command=W + "uv run src/bureau-elt/generate_bureau_files.py --date $DATE",
    )
    load = BashOperator(
        task_id="load",
        append_env=True,
        env=ENV,
        bash_command=W + "uv run src/bureau-elt/glue_bureau_job.py --date $DATE",
    )
    dbt_run = BashOperator(
        task_id="dbt_run",
        append_env=True,
        env=ENV,
        bash_command=W
        + CERTS
        + DBT
        + " run --project-dir src/bureau-elt/dbt --target-path /tmp/dbt_t --log-path /tmp/dbt_l",
    )
    dbt_test = BashOperator(
        task_id="dbt_test",
        append_env=True,
        env=ENV,
        bash_command=W
        + CERTS
        + DBT
        + " test --project-dir src/bureau-elt/dbt --target-path /tmp/dbt_t --log-path /tmp/dbt_l",
    )
    features = BashOperator(
        task_id="features",
        append_env=True,
        env=ENV,
        bash_command=W + "uv run src/feature-store/ingest_features.py",
    )
    seed >> load >> dbt_run >> dbt_test >> features
