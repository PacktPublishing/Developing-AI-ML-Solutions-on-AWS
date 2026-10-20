"""Who asked what, and what the answer cited: one JSON line per request.

On AWS the line lands in the app's CloudWatch log group, which keeps it for the
retention the template sets. Locally it prints to the terminal.
"""

import base64
import json
import os
from datetime import UTC, datetime
from functools import cache

import jwt


@cache
def _keys(issuer: str) -> jwt.PyJWKClient:
    """Return a client for the user pool's published signing keys."""
    return jwt.PyJWKClient(f"{issuer}/.well-known/jwks.json")


def _verified(token: str, issuer: str) -> dict:
    """Return the claims of an ID token the user pool signed for this app."""
    key = _keys(issuer).get_signing_key_from_jwt(token).key
    claims = jwt.decode(
        token,
        key,
        algorithms=["RS256"],
        issuer=issuer,
        audience=os.environ["COGNITO_CLIENT_ID"],
    )
    if claims.get("token_use") != "id":
        raise jwt.InvalidTokenError("not an ID token")
    return claims


def underwriter(authorization: str | None) -> str:
    """Return the signed-in underwriter's username.

    With COGNITO_ISSUER set, the app checks the token itself: the signature
    against the pool's keys, the issuer, the audience, and the expiry. API
    Gateway checks the same on AWS, and locally nothing else does. Without it
    the app runs open, and a request with no token is "local".

    Raises PermissionError when a token is required and is missing or invalid.
    """
    issuer = os.environ.get("COGNITO_ISSUER")
    token = (authorization or "").removeprefix("Bearer ").strip()
    if issuer:
        if not token:
            raise PermissionError("sign in first")
        try:
            claims = _verified(token, issuer)
        except jwt.PyJWTError as err:
            raise PermissionError(str(err)) from err
    elif not token:
        return "local"
    else:
        payload = token.split(".")[1]
        claims = json.loads(
            base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4))
        )
    return claims.get("cognito:username") or claims.get("username") or claims["sub"]


def record(
    who: str, route: str, text: str, cited: list, filters: dict | None = None
) -> dict:
    """Print and return the audit line for one request."""
    line = {
        "audit": True,
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
        "underwriter": who,
        "route": route,
        "text": text,
        "filters": {k: v for k, v in (filters or {}).items() if v is not None},
        "cited": cited,
    }
    print(json.dumps(line), flush=True)
    return line
