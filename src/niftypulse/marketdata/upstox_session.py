"""Upstox OAuth login and access-token storage.

Upstox tokens expire at 03:30 IST every day, so the operator logs in once each
morning. The redirect URL points at localhost over https with nothing listening, so the
browser shows an error page after login; the operator pastes that URL back and this
module checks the CSRF state and exchanges the one-time code for a token.

The token is written to a file outside the repo, never to .env or the logs.
"""

import secrets
from datetime import datetime, time, timedelta
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse
from zoneinfo import ZoneInfo

import httpx
import msgspec

from niftypulse.core.config import BrokerSettings

IST = ZoneInfo("Asia/Kolkata")
_DAILY_EXPIRY = time(3, 30)


class AuthError(RuntimeError):
    pass


class StoredToken(msgspec.Struct, frozen=True, kw_only=True):
    access_token: str
    user_id: str
    issued_at: datetime
    expires_at: datetime

    def is_valid(self, now: datetime | None = None) -> bool:
        return (now or datetime.now(IST)) < self.expires_at


def next_expiry(issued: datetime) -> datetime:
    expiry = datetime.combine(issued.date(), _DAILY_EXPIRY, tzinfo=IST)
    return expiry if issued < expiry else expiry + timedelta(days=1)


class UpstoxSession:
    def __init__(self, settings: BrokerSettings) -> None:
        self._s = settings
        self._token_path = Path(settings.token_file)

    def authorization_url(self) -> tuple[str, str]:
        state = secrets.token_urlsafe(24)
        query = urlencode(
            {
                "response_type": "code",
                "client_id": self._s.api_key.get_secret_value(),
                "redirect_uri": self._s.redirect_uri,
                "state": state,
            }
        )
        return f"{self._s.api_base_url}/v2/login/authorization/dialog?{query}", state

    @staticmethod
    def code_from_redirect(redirect_url: str, expected_state: str) -> str:
        params = parse_qs(urlparse(redirect_url.strip()).query)
        state = params.get("state", [""])[0]
        if not secrets.compare_digest(state, expected_state):
            raise AuthError("state mismatch: the pasted URL is not from this login attempt")
        code = params.get("code", [""])[0]
        if not code:
            raise AuthError("no authorization code in the pasted URL")
        return code

    def exchange(self, code: str) -> StoredToken:
        resp = httpx.post(
            f"{self._s.api_base_url}/v2/login/authorization/token",
            data={
                "code": code,
                "client_id": self._s.api_key.get_secret_value(),
                "client_secret": self._s.api_secret.get_secret_value(),
                "redirect_uri": self._s.redirect_uri,
                "grant_type": "authorization_code",
            },
            headers={"Accept": "application/json"},
            timeout=self._s.http_timeout_s,
        )
        if resp.status_code != httpx.codes.OK:
            raise AuthError(f"token exchange failed with HTTP {resp.status_code}")
        body = resp.json()
        issued = datetime.now(IST)
        token = StoredToken(
            access_token=body["access_token"],
            user_id=body.get("user_id", ""),
            issued_at=issued,
            expires_at=next_expiry(issued),
        )
        self._save(token)
        return token

    def load(self) -> StoredToken:
        if not self._token_path.exists():
            raise AuthError("no Upstox token found; run `niftypulse login`")
        token = msgspec.json.decode(self._token_path.read_bytes(), type=StoredToken)
        if not token.is_valid():
            raise AuthError(f"Upstox token expired at {token.expires_at:%Y-%m-%d %H:%M} IST")
        return token

    def _save(self, token: StoredToken) -> None:
        self._token_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._token_path.with_suffix(".tmp")
        tmp.write_bytes(msgspec.json.encode(token))
        tmp.chmod(0o600)
        tmp.replace(self._token_path)
