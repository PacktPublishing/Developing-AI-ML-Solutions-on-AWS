# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["boto3", "requests"]
# ///
"""Get loan decisions from the gateway running on the EKS cluster.

The EKS path has no ALB or Ingress (eksctl does not install the load balancer
controller), so this reaches the in-cluster gateway Service through a kubectl
port-forward, POSTs applications to /decision with requests, and reads the rows
the pod wrote to the DynamoDB log with boto3.

Usage: uv run aws/eks/decide.py                 # the three canonical applications
       uv run aws/eks/decide.py '<application-json>'  # one application of your own
"""

import json
import subprocess
import sys
import time

import boto3
import requests

CLUSTER = "ch04-decision"
REGION = "us-east-1"
TABLE = "decision_log"
PORT = 8085

APPLICATIONS = [
    {
        "application_id": "app-1042",
        "age": 40,
        "monthly_income": 6000,
        "requested_amount": 5000,
        "dti": 0.10,
        "utilization": 0.10,
    },
    {
        "application_id": "app-1043",
        "age": 34,
        "monthly_income": 4200,
        "requested_amount": 8000,
        "dti": 0.20,
        "utilization": 0.20,
    },
    {
        "application_id": "app-1044",
        "age": 34,
        "monthly_income": 4200,
        "requested_amount": 8000,
        "dti": 0.20,
        "utilization": 0.15,
        "kyc_passed": False,
    },
]


def decide(base_url: str, application: dict) -> dict:
    """POST one application to the gateway, retrying the serverless endpoints' cold-start 5xx."""
    for attempt in range(5):
        resp = requests.post(f"{base_url}/decision", json=application, timeout=30)
        if resp.status_code < 500:
            resp.raise_for_status()
            return resp.json()
        time.sleep(5)  # a model endpoint was still warming; give it a moment
    resp.raise_for_status()
    return resp.json()


def main() -> None:
    """Port-forward to the gateway, send applications, and show the logged rows."""
    # Point kubectl at the cluster, then wait for the gateway pod to be ready.
    subprocess.run(
        ["aws", "eks", "update-kubeconfig", "--name", CLUSTER, "--region", REGION],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    subprocess.run(
        [
            "kubectl",
            "wait",
            "--for=condition=ready",
            "pod",
            "-l",
            "app=gateway",
            "--timeout=180s",
        ],
        check=True,
    )

    # Forward a local port to the in-cluster gateway Service (no Ingress on EKS here).
    forward = subprocess.Popen(
        ["kubectl", "port-forward", "svc/gateway", f"{PORT}:80"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        base_url = f"http://localhost:{PORT}"
        for _ in range(20):  # wait for the tunnel to accept connections
            try:
                if requests.get(f"{base_url}/health", timeout=1).ok:
                    break
            except requests.RequestException:
                time.sleep(1)

        applications = [json.loads(sys.argv[1])] if len(sys.argv) > 1 else APPLICATIONS
        for application in applications:
            d = decide(base_url, application)
            print(
                f"{d['decision']:8} pd={d['pd']:.4f}  models={d['model_pds']}  reasons={d['reasons']}"
            )
    finally:
        forward.terminate()

    # The persisted results: the decisions the gateway logged to DynamoDB.
    count = (
        boto3.resource("dynamodb", region_name=REGION)
        .Table(TABLE)
        .scan(Select="COUNT")["Count"]
    )
    print("decision_log rows:", count)


if __name__ == "__main__":
    main()
