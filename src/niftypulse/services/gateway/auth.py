"""Authentication, authorisation and rate limiting for the API gateway.

- API keys look like `<principal>.<secret>`. Only an scrypt hash of the secret is stored;
  verification is constant-time.
- A key is exchanged for a short-lived HS256 JWT carrying the principal and role.
- Roles are ordered viewer < operator < admin; each route declares the minimum role.
- A token bucket per principal (or client IP before login) limits request rates.
"""

import base64
import hashlib
import hmac
import secrets
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from niftypulse.core.config import GatewaySettings

ROLES = {"viewer": 0, "operator": 1, "admin": 2}
_SCRYPT = {"n": 2**14, "r": 8, "p": 1}


def new_api_key(principal: str) -> tuple[str, str]:
    """(plaintext key shown once, stored hash)."""
    secret = secrets.token_urlsafe(32)
    return f"{principal}.{secret}", hash_secret(secret)


def hash_secret(secret: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(secret.encode(), salt=salt, dklen=32, **_SCRYPT)
    return "scrypt$" + base64.b64encode(salt).decode() + "$" + base64.b64encode(digest).decode()


def verify_secret(secret: str, stored: str) -> bool:
    try:
        scheme, salt_b64, digest_b64 = stored.split("$")
    except ValueError:
        return False
    if scheme != "scrypt":
        return False
    digest = hashlib.scrypt(secret.encode(), salt=base64.b64decode(salt_b64), dklen=32, **_SCRYPT)
    return hmac.compare_digest(digest, base64.b64decode(digest_b64))


@dataclass(frozen=True, slots=True)
class Principal:
    id: str
    role: str


def issue_token(principal: Principal, s: GatewaySettings) -> str:
    now = int(time.time())
    claims = {
        "sub": principal.id,
        "role": principal.role,
        "iat": now,
        "exp": now + s.jwt_ttl_s,
        "jti": uuid.uuid4().hex,
    }
    return jwt.encode(claims, s.jwt_secret.get_secret_value(), algorithm="HS256")


def decode_token(token: str, s: GatewaySettings) -> Principal:
    try:
        claims = jwt.decode(
            token,
            s.jwt_secret.get_secret_value(),
            algorithms=["HS256"],
            options={"require": ["exp", "iat", "sub"]},
        )
    except jwt.PyJWTError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or expired token") from exc
    role = claims.get("role")
    if role not in ROLES:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid token role")
    return Principal(str(claims["sub"]), str(role))


class RateLimiter:
    """Token bucket per key; in-process (one gateway instance)."""

    def __init__(self, per_minute: int) -> None:
        self._rate = per_minute / 60.0
        self._capacity = float(per_minute)
        self._buckets: dict[str, tuple[float, float]] = {}

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        tokens, last = self._buckets.get(key, (self._capacity, now))
        tokens = min(self._capacity, tokens + (now - last) * self._rate)
        if tokens < 1.0:
            self._buckets[key] = (tokens, now)
            return False
        self._buckets[key] = (tokens - 1.0, now)
        return True


_bearer = HTTPBearer(auto_error=False)


def require(min_role: str) -> Callable[..., Awaitable[Principal]]:
    async def dependency(
        request: Request, creds: HTTPAuthorizationCredentials | None = Depends(_bearer)
    ) -> Principal:
        if creds is None:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing bearer token")
        principal = decode_token(creds.credentials, request.app.state.settings)
        if not request.app.state.limiter.allow(principal.id):
            raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "rate limit exceeded")
        if ROLES[principal.role] < ROLES[min_role]:
            raise HTTPException(status.HTTP_403_FORBIDDEN, f"requires role {min_role}")
        request.state.principal = principal
        return principal

    return dependency
