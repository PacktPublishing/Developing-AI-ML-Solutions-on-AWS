# /// script
# dependencies = ["boto3"]
# ///
"""Create an underwriter in the user pool, and call the deployed app as one.

The password is generated into Secrets Manager and read back from there, so it is
never typed, echoed, or passed on a command line. With COGNITO_ENDPOINT set, the
same commands run against the local Cognito (local/cognito) and the local app
instead, with the pool and client ids and the password from the environment.

Usage:
  uv run --no-project underwriter.py add
  uv run --no-project underwriter.py call ask '{"query": "How is DTI assessed?"}'
  uv run --no-project underwriter.py call ask '{"query": "..."}' --no-token
"""

import argparse
import os
import urllib.error
import urllib.request

import boto3

REGION = "us-east-1"
STACK = "ch09-underwriting-kb"
SECRET = "ch09-underwriter"
USERNAME = "underwriter1"
LOCAL = os.environ.get("COGNITO_ENDPOINT")


def outputs() -> dict[str, str]:
    """Return the app stack's outputs by key."""
    if LOCAL:
        return {
            "UserPoolId": os.environ["COGNITO_POOL_ID"],
            "UserPoolClientId": os.environ["COGNITO_CLIENT_ID"],
            "AppUrl": os.environ.get("APP_URL", "http://localhost:8080"),
        }
    cfn = boto3.client("cloudformation", region_name=REGION)
    stack = cfn.describe_stacks(StackName=STACK)["Stacks"][0]
    return {o["OutputKey"]: o["OutputValue"] for o in stack["Outputs"]}


def password() -> str:
    """Return the underwriter's password, generating it on first use."""
    if LOCAL:
        return os.environ["UNDERWRITER_PASSWORD"]
    sm = boto3.client("secretsmanager", region_name=REGION)
    try:
        return sm.get_secret_value(SecretId=SECRET)["SecretString"]
    except sm.exceptions.ResourceNotFoundException:
        # the pool demands a symbol, so add one the generator was told to leave out
        pw = sm.get_random_password(PasswordLength=20, ExcludePunctuation=True)
        value = pw["RandomPassword"] + "aA1!"
        sm.create_secret(Name=SECRET, SecretString=value)
        return value


def cognito():
    """Return a client for the user pool: cognito-shim locally, Cognito on AWS."""
    if LOCAL:
        # the local pool checks no credentials, so none of the real ones are sent
        return boto3.client(
            "cognito-idp",
            region_name=REGION,
            endpoint_url=LOCAL,
            aws_access_key_id="local",
            aws_secret_access_key="local",
        )
    return boto3.client("cognito-idp", region_name=REGION)


def add() -> None:
    """Create the underwriter with a permanent password, so sign-in needs no reset."""
    pool = outputs()["UserPoolId"]
    idp = cognito()
    try:
        idp.admin_create_user(
            UserPoolId=pool, Username=USERNAME, MessageAction="SUPPRESS"
        )
    except idp.exceptions.UsernameExistsException:
        pass
    idp.admin_set_user_password(
        UserPoolId=pool, Username=USERNAME, Password=password(), Permanent=True
    )
    print(f"underwriter {USERNAME} ready in {pool}")


def token(client_id: str) -> str:
    """Sign in as the underwriter and return the ID token API Gateway checks."""
    idp = cognito()
    resp = idp.initiate_auth(
        ClientId=client_id,
        AuthFlow="USER_PASSWORD_AUTH",
        AuthParameters={"USERNAME": USERNAME, "PASSWORD": password()},
    )
    return resp["AuthenticationResult"]["IdToken"]


def call(route: str, body: str, signed: bool) -> None:
    """POST a JSON body to one route and print the status and the reply."""
    out = outputs()
    headers = {"Content-Type": "application/json"}
    if signed:
        headers["Authorization"] = f"Bearer {token(out['UserPoolClientId'])}"
    req = urllib.request.Request(
        f"{out['AppUrl']}/{route}", data=body.encode(), headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            status, reply = resp.status, resp.read().decode()
    except urllib.error.HTTPError as err:
        status, reply = err.code, err.read().decode()
    print(status)
    print(reply)


def main() -> None:
    """Parse the subcommand and run it."""
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("add", help="create the underwriter")
    c = sub.add_parser("call", help="call a route as the underwriter")
    c.add_argument("route")
    c.add_argument("body")
    c.add_argument("--no-token", action="store_true", help="call without signing in")
    a = p.parse_args()
    if a.cmd == "add":
        add()
    else:
        call(a.route, a.body, not a.no_token)


if __name__ == "__main__":
    main()
