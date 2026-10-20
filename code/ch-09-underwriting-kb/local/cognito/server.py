"""A from-source Cognito user pool for the local stack: the operations ch-09 uses.

AdminCreateUser and AdminSetUserPassword (underwriter.py add), InitiateAuth with
USER_PASSWORD_AUTH (the page's sign-in), and the pool's JWKS (the app's token
check). The pool's password policy and the client's auth flows are read from the
AWS::Cognito::* resources of aws/template.yaml, so they are the template's own.
Tokens are RS256 JWTs in Cognito's shape, signed with a key made at startup.
Users live in memory, so a restart clears them.
"""

import base64
import json
import os
import re
import secrets
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import yaml
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa

POOL_ID = os.environ.get("COGNITO_POOL_ID", "us-east-1_ch09Local")
CLIENT_ID = os.environ.get("COGNITO_CLIENT_ID", "ch09underwriterapplocal000")
PUBLIC_URL = os.environ.get("PUBLIC_URL", "http://localhost:9229")
ISSUER = f"{PUBLIC_URL}/{POOL_ID}"
KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
KID = secrets.token_hex(8)
USERS: dict[str, dict] = {}


class CognitoError(Exception):
    """An error botocore and the page read the way they read Cognito's."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


def _template_loader() -> type[yaml.SafeLoader]:
    """Return a YAML loader that reads CloudFormation's !Ref, !Sub, and the rest."""

    class Loader(yaml.SafeLoader):
        pass

    def tagged(loader, suffix, node):
        """Keep the value under a short-form intrinsic and drop the tag."""
        if isinstance(node, yaml.ScalarNode):
            return loader.construct_scalar(node)
        if isinstance(node, yaml.SequenceNode):
            return loader.construct_sequence(node)
        return loader.construct_mapping(node)

    Loader.add_multi_constructor("!", tagged)
    return Loader


def load_pool(path: str) -> tuple[dict, list[str]]:
    """Return the template's password policy and the client's explicit auth flows."""
    with open(path) as fh:
        resources = yaml.load(fh, Loader=_template_loader())["Resources"]
    by_type = {r["Type"]: r.get("Properties", {}) for r in resources.values()}
    policy = by_type["AWS::Cognito::UserPool"]["Policies"]["PasswordPolicy"]
    flows = by_type["AWS::Cognito::UserPoolClient"]["ExplicitAuthFlows"]
    return policy, flows


POLICY, FLOWS = load_pool(os.environ.get("TEMPLATE", "/stack/template.yaml"))


def check_password(password: str) -> None:
    """Raise InvalidPasswordException if the password breaks the pool's policy."""
    rules = [
        ("RequireLowercase", r"[a-z]", "Password must have lowercase characters"),
        ("RequireUppercase", r"[A-Z]", "Password must have uppercase characters"),
        ("RequireNumbers", r"[0-9]", "Password must have numeric characters"),
        ("RequireSymbols", r"[^A-Za-z0-9]", "Password must have symbol characters"),
    ]
    if len(password) < int(POLICY.get("MinimumLength", 8)):
        raise CognitoError(
            "InvalidPasswordException",
            "Password did not conform with policy: Password not long enough",
        )
    for flag, pattern, message in rules:
        if POLICY.get(flag) and not re.search(pattern, password):
            raise CognitoError(
                "InvalidPasswordException",
                f"Password did not conform with policy: {message}",
            )


def _b64(data: bytes) -> str:
    """Return data as unpadded base64url, the encoding JWTs use."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def sign(claims: dict) -> str:
    """Return the claims as an RS256 JWT signed with the pool's key."""
    header = _b64(json.dumps({"kid": KID, "alg": "RS256"}).encode())
    payload = _b64(json.dumps(claims).encode())
    signature = KEY.sign(
        f"{header}.{payload}".encode(), padding.PKCS1v15(), hashes.SHA256()
    )
    return f"{header}.{payload}.{_b64(signature)}"


def jwks() -> dict:
    """Return the pool's public key as the JWKS document Cognito publishes."""
    numbers = KEY.public_key().public_numbers()
    return {
        "keys": [
            {
                "kid": KID,
                "alg": "RS256",
                "kty": "RSA",
                "use": "sig",
                "n": _b64(numbers.n.to_bytes((numbers.n.bit_length() + 7) // 8)),
                "e": _b64(numbers.e.to_bytes(3)),
            }
        ]
    }


def _pool(body: dict) -> None:
    """Raise ResourceNotFoundException unless the request names this pool."""
    if body.get("UserPoolId") != POOL_ID:
        raise CognitoError("ResourceNotFoundException", "User pool does not exist.")


def admin_create_user(body: dict) -> dict:
    """Create a user who must change a temporary password, as Cognito does."""
    _pool(body)
    name = body["Username"]
    if name in USERS:
        raise CognitoError("UsernameExistsException", "User account already exists")
    USERS[name] = {
        "sub": str(uuid.uuid4()),
        "password": body.get("TemporaryPassword") or secrets.token_urlsafe(12),
        "status": "FORCE_CHANGE_PASSWORD",
    }
    return {"User": {"Username": name, "UserStatus": "FORCE_CHANGE_PASSWORD"}}


def admin_set_user_password(body: dict) -> dict:
    """Set a user's password, confirming the user when it is permanent."""
    _pool(body)
    user = USERS.get(body["Username"])
    if user is None:
        raise CognitoError("UserNotFoundException", "User does not exist.")
    check_password(body["Password"])
    user["password"] = body["Password"]
    if body.get("Permanent"):
        user["status"] = "CONFIRMED"
    return {}


def initiate_auth(body: dict) -> dict:
    """Sign a user in with USER_PASSWORD_AUTH and return ID and access tokens."""
    if body.get("ClientId") != CLIENT_ID:
        raise CognitoError(
            "ResourceNotFoundException", "User pool client does not exist."
        )
    flow = body.get("AuthFlow")
    if flow != "USER_PASSWORD_AUTH" or f"ALLOW_{flow}" not in FLOWS:
        raise CognitoError(
            "InvalidParameterException", f"{flow} flow not enabled for this client"
        )
    params = body.get("AuthParameters", {})
    user = USERS.get(params.get("USERNAME", ""))
    if user is None or not secrets.compare_digest(
        user["password"], params.get("PASSWORD", "")
    ):
        raise CognitoError("NotAuthorizedException", "Incorrect username or password.")
    if user["status"] != "CONFIRMED":
        raise CognitoError(
            "NotAuthorizedException", "Password change required; set it as an admin."
        )
    now = int(time.time())
    session = {
        "sub": user["sub"],
        "iss": ISSUER,
        "auth_time": now,
        "iat": now,
        "exp": now + 3600,
        "origin_jti": str(uuid.uuid4()),
        "event_id": str(uuid.uuid4()),
    }
    username = params["USERNAME"]
    id_token = sign(
        {
            **session,
            "aud": CLIENT_ID,
            "token_use": "id",
            "cognito:username": username,
            "jti": str(uuid.uuid4()),
        }
    )
    access_token = sign(
        {
            **session,
            "client_id": CLIENT_ID,
            "token_use": "access",
            "scope": "aws.cognito.signin.user.admin",
            "username": username,
            "jti": str(uuid.uuid4()),
        }
    )
    return {
        "ChallengeParameters": {},
        "AuthenticationResult": {
            "IdToken": id_token,
            "AccessToken": access_token,
            "RefreshToken": secrets.token_urlsafe(48),
            "ExpiresIn": 3600,
            "TokenType": "Bearer",
        },
    }


OPERATIONS = {
    "AdminCreateUser": admin_create_user,
    "AdminSetUserPassword": admin_set_user_password,
    "InitiateAuth": initiate_auth,
}


class Handler(BaseHTTPRequestHandler):
    """Serve the cognito-idp operations and the JWKS the chapter calls."""

    def do_OPTIONS(self):
        """Answer the browser's preflight before the page's sign-in POST."""
        self._send(200, {})

    def do_GET(self):
        """Serve the pool's JWKS, and the health check on any other path."""
        if self.path == f"/{POOL_ID}/.well-known/jwks.json":
            self._send(200, jwks())
        else:
            self._send(200, {"ok": True})

    def do_POST(self):
        """Dispatch one cognito-idp operation by its X-Amz-Target."""
        target = self.headers.get("X-Amz-Target", "").split(".")[-1]
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
        operation = OPERATIONS.get(target)
        try:
            if operation is None:
                raise CognitoError(
                    "InvalidAction", f"{target} is not supported locally"
                )
            self._send(200, operation(body))
        except CognitoError as err:
            self._send(400, {"__type": err.kind, "message": str(err)}, err.kind)

    def _send(self, code: int, payload: dict, error: str | None = None) -> None:
        """Write a JSON reply with the CORS headers the page needs."""
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/x-amz-json-1.1")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "content-type, x-amz-target")
        if error:
            self.send_header("x-amzn-ErrorType", error)
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args) -> None:
        """Keep the server quiet."""


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", 9229), Handler).serve_forever()
