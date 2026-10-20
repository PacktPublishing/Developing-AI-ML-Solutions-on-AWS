"""The app accepts only an ID token the user pool signed for it."""

import time
from types import SimpleNamespace

import audit
import jwt
import pytest
from audit import underwriter
from cryptography.hazmat.primitives.asymmetric import rsa

ISSUER = "http://localhost:9229/us-east-1_test"
CLIENT = "client-under-test"
POOL_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _token(key=POOL_KEY, **claims) -> str:
    """Return a bearer header for an ID token with these claims over the defaults."""
    now = int(time.time())
    body = {
        "iss": ISSUER,
        "aud": CLIENT,
        "token_use": "id",
        "cognito:username": "underwriter1",
        "sub": "abc",
        "iat": now,
        "exp": now + 300,
        **claims,
    }
    return "Bearer " + jwt.encode(body, key, algorithm="RS256")


@pytest.fixture(autouse=True)
def pool(monkeypatch):
    """Require tokens, and serve the pool's public key instead of fetching JWKS."""
    monkeypatch.setenv("COGNITO_ISSUER", ISSUER)
    monkeypatch.setenv("COGNITO_CLIENT_ID", CLIENT)
    keys = SimpleNamespace(
        get_signing_key_from_jwt=lambda _: SimpleNamespace(key=POOL_KEY.public_key())
    )
    monkeypatch.setattr(audit, "_keys", lambda _: keys)


def test_a_token_from_the_pool_names_the_underwriter():
    """A valid ID token yields the username the audit line records."""
    assert underwriter(_token()) == "underwriter1"


@pytest.mark.parametrize(
    "authorization",
    [
        None,
        _token(key=rsa.generate_private_key(public_exponent=65537, key_size=2048)),
        _token(aud="another-app"),
        _token(iss="https://cognito-idp.us-east-1.amazonaws.com/other"),
        _token(exp=int(time.time()) - 60),
        _token(token_use="access"),
    ],
    ids=["missing", "forged", "audience", "issuer", "expired", "access-token"],
)
def test_anything_else_is_refused(authorization):
    """No token, another key, app, or pool, an expired token, or an access token."""
    with pytest.raises(PermissionError):
        underwriter(authorization)
