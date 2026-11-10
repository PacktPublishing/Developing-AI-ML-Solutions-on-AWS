"""Start the DynamoDB Local that the Feature Store shim needs.

app.py builds a FeatureStore when it is imported, and that connects to DynamoDB
and creates the metadata table, so the store has to be up before the test module
is imported at all. pytest reads this file first, which is what lets the test
module keep every import at the top.
"""

import atexit
import os
import socket
import time

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

DEFAULT_URL = "http://localhost:8000"
# Pinned: `latest` moves under the test, and the shim reads a fixed API shape.
IMAGE = "amazon/dynamodb-local:3.3.2"

# Nothing to test against if no store can be reached; pytest reads this list.
collect_ignore: list[str] = []


def _dynamodb_ready(url: str) -> bool:
    """Ask DynamoDB to list its tables and check that the reply is one.

    Port 8000 is popular, and a probe that only opens a socket cannot tell
    DynamoDB Local from whatever else the host happens to be running: it reports
    ready, the container is never started, and the first call gets a connection
    reset. Reading TableNames out of the reply is the check, because a foreign
    server answering with an HTTP error raises ClientError just as a real one
    would.
    """
    try:
        reply = boto3.client(
            "dynamodb",
            endpoint_url=url,
            region_name="us-east-1",
            aws_access_key_id="local",
            aws_secret_access_key="localsecret",
            config=Config(
                retries={"max_attempts": 0}, connect_timeout=2, read_timeout=2
            ),
        ).list_tables()
    except (ClientError, BotoCoreError, OSError):
        return False
    return "TableNames" in reply


def _free_port() -> int:
    """Take a port the host is not using, so a busy 8000 cannot collide."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _start_dynamodb(port: int) -> bool:
    """Start DynamoDB Local on a port, stop it when tests end, say whether it came up."""
    import docker

    try:
        engine = docker.from_env()
    except docker.errors.DockerException:
        return False
    container = engine.containers.run(
        IMAGE,
        command=["-jar", "DynamoDBLocal.jar", "-inMemory"],
        ports={"8000/tcp": port},
        detach=True,
        remove=True,
    )
    atexit.register(container.stop)
    for _ in range(30):
        if _dynamodb_ready(f"http://localhost:{port}"):
            return True
        time.sleep(1)
    return False


# Reuse a store the developer already has up, which is what local/lakehouse.yml
# publishes on 8000, and start one otherwise. The shim reads ONLINE_STORE_URL.
STORE_URL = os.environ.get("ONLINE_STORE_URL", DEFAULT_URL)
if not _dynamodb_ready(STORE_URL):
    OWN_PORT = _free_port()
    STORE_URL = f"http://localhost:{OWN_PORT}" if _start_dynamodb(OWN_PORT) else ""

if STORE_URL:
    os.environ["ONLINE_STORE_URL"] = STORE_URL
else:
    print(
        "no DynamoDB Local and no Docker daemon to start one; skipping the shim tests"
    )
    collect_ignore.append("test_sagemaker_local.py")
