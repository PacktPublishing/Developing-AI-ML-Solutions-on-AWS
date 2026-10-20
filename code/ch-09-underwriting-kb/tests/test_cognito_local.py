"""The local user pool signs underwriters in the way the app expects from Cognito."""

import importlib.util
import socket
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import boto3
import pytest
from audit import underwriter

pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parent.parent
PASSWORD = "Local-underwriter-1!"


def _free_port() -> int:
    """Return a port nothing is listening on."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def pool(tmp_path_factory):
    """Start the local pool on a free port, read from the chapter's template."""
    port = _free_port()
    mp = pytest.MonkeyPatch()
    mp.setenv("TEMPLATE", str(ROOT / "aws" / "template.yaml"))
    mp.setenv("PUBLIC_URL", f"http://127.0.0.1:{port}")
    spec = importlib.util.spec_from_file_location(
        "cognito_server", ROOT / "local" / "cognito" / "server.py"
    )
    assert spec and spec.loader
    server = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(server)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), server.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    idp = boto3.client(
        "cognito-idp",
        region_name="us-east-1",
        endpoint_url=f"http://127.0.0.1:{port}",
        aws_access_key_id="local",
        aws_secret_access_key="local",
    )
    yield server, idp
    httpd.shutdown()
    mp.undo()


def _sign_in(server, idp, password: str = PASSWORD) -> dict:
    """Return the AuthenticationResult for underwriter1."""
    return idp.initiate_auth(
        ClientId=server.CLIENT_ID,
        AuthFlow="USER_PASSWORD_AUTH",
        AuthParameters={"USERNAME": "underwriter1", "PASSWORD": password},
    )["AuthenticationResult"]


def test_the_policy_is_the_templates(pool):
    """The password policy and auth flows come from aws/template.yaml."""
    server, _ = pool
    assert server.POLICY["MinimumLength"] == 12
    assert "ALLOW_USER_PASSWORD_AUTH" in server.FLOWS


def test_underwriter_signs_in_and_the_app_accepts_the_token(pool, monkeypatch):
    """Create, set a permanent password, sign in, and pass the app's own check."""
    server, idp = pool
    idp.admin_create_user(
        UserPoolId=server.POOL_ID, Username="underwriter1", MessageAction="SUPPRESS"
    )
    with pytest.raises(idp.exceptions.NotAuthorizedException):
        _sign_in(server, idp)  # still on the temporary password
    with pytest.raises(idp.exceptions.InvalidPasswordException):
        idp.admin_set_user_password(
            UserPoolId=server.POOL_ID,
            Username="underwriter1",
            Password="short",
            Permanent=True,
        )
    idp.admin_set_user_password(
        UserPoolId=server.POOL_ID,
        Username="underwriter1",
        Password=PASSWORD,
        Permanent=True,
    )
    tokens = _sign_in(server, idp)
    monkeypatch.setenv("COGNITO_ISSUER", server.ISSUER)
    monkeypatch.setenv("COGNITO_CLIENT_ID", server.CLIENT_ID)
    assert underwriter(f"Bearer {tokens['IdToken']}") == "underwriter1"
    with pytest.raises(PermissionError):
        underwriter(f"Bearer {tokens['AccessToken']}")


def test_a_wrong_password_is_refused(pool):
    """The error is the one botocore raises for Cognito."""
    server, idp = pool
    with pytest.raises(idp.exceptions.NotAuthorizedException):
        _sign_in(server, idp, password="Wrong-password-1!")
