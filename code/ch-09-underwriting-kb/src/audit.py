"""Who asked what, and what the answer cited: one JSON line per request.

On AWS the line lands in the app's CloudWatch log group, which keeps it for the
retention the template sets. Locally it prints to the terminal.
"""

import base64
import json
from datetime import UTC, datetime


def underwriter(authorization: str | None) -> str:
    """Return the signed-in underwriter's username, or "local" with no token.

    API Gateway's JWT authorizer has already verified the token before the
    request reaches the app, so this only reads the claim it vouched for.
    """
    if not authorization or not authorization.startswith("Bearer "):
        return "local"
    payload = authorization.split()[1].split(".")[1]
    claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
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
