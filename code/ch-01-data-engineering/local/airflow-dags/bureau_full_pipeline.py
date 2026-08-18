"""The full nightly pipeline orchestrated by Airflow, end to end.

Stage the raw bureau files, load them with dlt, build the dbt marts, test them, then
ingest the result into the feature store (Iceberg offline + DynamoDB online). The same
scripts `make run/dbt-run/features` run, now wired as a DAG against the lakehouse stack.
"""

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
    "SSL_CERT_FILE": "/work/oblako-bundle.pem",
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
DBT = "uvx --from dbt-core==1.12.2 --with dbt-redshift==1.11.0 dbt"
W = "set -e; cd /work; "

with DAG(
    "bureau_full_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule=None,
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
        + DBT
        + " run --project-dir src/bureau-elt/dbt --target-path /tmp/dbt_t --log-path /tmp/dbt_l",
    )
    dbt_test = BashOperator(
        task_id="dbt_test",
        append_env=True,
        env=ENV,
        bash_command=W
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
