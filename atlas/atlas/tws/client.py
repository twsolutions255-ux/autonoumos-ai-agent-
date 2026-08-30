"""Async HTTP client for the TW Solutions API.

This is the only place in Atlas that knows how to talk to the app. Everything
else — tools, brain, console — goes through it, which means authentication,
retries, rate limiting and the audit trail are implemented exactly once.

Design notes worth keeping:

* **The token is refreshed, never assumed.** TWS issues a 7-day JWT and
  invalidates it early if the password changes (`password_changed_at` is
  stamped into the token as `pwd`). A long-running agent therefore *will* meet
  a 401 in normal operation, and must recover from it silently instead of
  going dark until someone restarts it.

* **Reads retry, writes do not.** A retried POST is a second cold call, a
  second email, a second charge. Idempotency is not something this API
  promises, so the client refuses to guess: only GETs are retried
  automatically, plus the explicit `retry_writes` opt-in for endpoints
  documented as safe.

* **Every call is offered to an audit sink.** Autonomy without a log is not
  something anyone should deploy.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Awaitable, Callable, Optional

import httpx

log = logging.getLogger("atlas.tws")

#: Statuses worth trying again. 401 is handled separately (re-login, then one
#: retry); 409/422 are the app telling us the request was wrong, and repeating
#: it unchanged just produces the same answer more slowly.
RETRY_STATUSES = {429, 500, 502, 503, 504}


class TWSError(RuntimeError):
    """A call to the app failed in a way the caller should see."""

    def __init__(self, message: str, *, status: Optional[int] = None,
                 method: str = "", path: str = "", body: Any = None):
        super().__init__(message)
        self.status = status
        self.method = method
        self.path = path
        self.body = body

    def as_tool_result(self) -> str:
        """Phrased for a model to read and act on, not for a stack trace."""
        where = f"{self.method} {self.path}".strip()
        if self.status == 403:
            return (f"Refused (403) by {where}. This account is not allowed to do that. "
                    f"Do not retry — pick a different approach or ask the owner.")
        if self.status == 404:
            return f"Not found (404) at {where}. The thing you referenced does not exist."
        if self.status == 400 or self.status == 422:
            return (f"Rejected ({self.status}) by {where}: {self.body}. "
                    f"Fix the arguments and try once more.")
        return f"Call to {where} failed{f' ({self.status})' if self.status else ''}: {self.args[0]}"


class TWSClient:
    """Authenticated, retrying, audited access to the TWS API."""

    def __init__(self, base_url: str, *, email: str = "", password: str = "",
                 token: str = "", cron_secret: str = "", timeout: float = 45.0,
                 audit: Optional[Callable[[dict], Awaitable[None]]] = None):
        if not base_url:
            raise ValueError("TWSClient needs a base_url")
        self.base_url = base_url.rstrip("/")
        # The app mounts everything under /api. Callers pass app-level paths
        # ("/admin/overview"); the prefix is this client's business.
        self.api_root = f"{self.base_url}/api"
        self._email = email
        self._password = password
        self._token = token
        self._cron_secret = cron_secret
        # True when the token came from the environment rather than a login we
        # performed. We cannot refresh such a token, so a 401 on it is fatal
        # and must say so rather than looping.
        self._token_is_static = bool(token and not (email and password))
        self._timeout = timeout
        self._audit = audit
        self._client: Optional[httpx.AsyncClient] = None
        self._login_lock = asyncio.Lock()
        self._me: Optional[dict] = None
        # The app locks an identifier for 15 minutes after 5 failed logins —
        # and it stays locked even for the CORRECT password. A wrong
        # TWS_PASSWORD would otherwise have Atlas lock its own account out
        # within five tool calls, then stay locked for as long as it runs.
        # So a failure here stops further attempts rather than retrying into
        # a self-inflicted outage.
        self._login_failures = 0
        self._login_blocked_until = 0.0

    # ---------- lifecycle ----------

    async def __aenter__(self) -> "TWSClient":
        await self.start()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    async def start(self) -> None:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self._timeout,
                # A single agent, but tools fan out concurrently within one turn.
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
                follow_redirects=True,
                headers={"User-Agent": "Atlas/1.0 (TWS autonomous scaling agent)"},
            )

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ---------- auth ----------

    async def login(self) -> str:
        """Exchange credentials for a bearer token. Safe to call concurrently."""
        async with self._login_lock:
            if self._token_is_static:
                raise TWSError(
                    "TWS_TOKEN was rejected and Atlas has no credentials to mint a new "
                    "one. Set TWS_EMAIL and TWS_PASSWORD so it can re-authenticate itself.",
                    status=401)
            if not (self._email and self._password):
                raise TWSError("No TWS credentials configured.", status=401)
            if time.time() < self._login_blocked_until:
                wait = int(self._login_blocked_until - time.time())
                raise TWSError(
                    f"Not attempting login for another {wait}s. {self._login_failures} "
                    f"attempts already failed, and the app locks an account for 15 minutes "
                    f"after 5 — continuing to try would lock Atlas out entirely. "
                    f"Check TWS_EMAIL/TWS_PASSWORD.",
                    status=401)
            await self.start()
            assert self._client is not None
            resp = await self._client.post(
                f"{self.api_root}/auth/login",
                json={"email": self._email, "password": self._password},
            )
            if resp.status_code != 200:
                self._login_failures += 1
                # Stop well short of the app's 5-strike lockout, and back off
                # hard so a transient outage does not become a cooldown loop.
                if self._login_failures >= 2:
                    self._login_blocked_until = time.time() + min(
                        900, 60 * (2 ** (self._login_failures - 2)))
                raise TWSError(
                    f"Login as {self._email} failed ({resp.status_code}), attempt "
                    f"{self._login_failures}. Atlas cannot reach the business until "
                    f"this is fixed.",
                    status=resp.status_code, method="POST", path="/auth/login",
                    body=_safe_body(resp))
            data = resp.json()
            # The app sets an httponly cookie AND returns the token in the body
            # on some paths. Accept either; prefer the explicit field.
            token = data.get("access_token") or data.get("token") or ""
            if not token:
                cookie = resp.cookies.get("access_token")
                token = cookie or ""
            if not token:
                raise TWSError(
                    "Login succeeded but returned no token Atlas could find "
                    "(checked body.access_token, body.token and the access_token cookie).",
                    status=200, method="POST", path="/auth/login")
            self._token = token
            self._login_failures = 0
            self._login_blocked_until = 0.0
            log.info("atlas: authenticated to TWS as %s", self._email)
            return token

    async def whoami(self, *, refresh: bool = False) -> dict:
        """The account Atlas is acting as. Cached — it does not change mid-run."""
        if self._me is None or refresh:
            self._me = await self.get("/auth/me")
        return self._me

    async def verify_access(self) -> dict:
        """Boot check: can Atlas log in, and is the account privileged enough?

        Called once at startup so a misconfigured deploy fails loudly and
        immediately rather than on the first write, hours later.
        """
        me = await self.whoami(refresh=True)
        roles = set(filter(None, [me.get("role")] + list(me.get("roles") or [])))
        return {
            "id": me.get("id"),
            "email": me.get("email"),
            "name": me.get("name"),
            "role": me.get("role"),
            "roles": sorted(roles),
            "is_superadmin": me.get("role") == "superadmin",
            "note": (
                "Atlas is superadmin and can drive the whole app."
                if me.get("role") == "superadmin" else
                "Atlas is NOT superadmin. Admin endpoints will refuse it (403); "
                "it will be able to read and chat but not run the business."
            ),
        }

    # ---------- the one request path ----------

    async def request(self, method: str, path: str, *, params: Optional[dict] = None,
                      json_body: Optional[dict] = None, retries: int = 2,
                      retry_writes: bool = False, expect_json: bool = True) -> Any:
        await self.start()
        assert self._client is not None
        method = method.upper()
        url = f"{self.api_root}{path if path.startswith('/') else '/' + path}"
        idempotent = method in ("GET", "HEAD") or retry_writes
        attempts = (retries + 1) if idempotent else 1

        if not self._token and not self._token_is_static:
            await self.login()

        started = time.monotonic()
        last_exc: Optional[Exception] = None
        relogged = False

        for attempt in range(attempts):
            try:
                resp = await self._client.request(
                    method, url, params=params, json=json_body,
                    headers={"Authorization": f"Bearer {self._token}"} if self._token else {},
                )
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last_exc = e
                if attempt + 1 < attempts:
                    await asyncio.sleep(2 ** attempt)
                    continue
                await self._log(method, path, None, started, str(e))
                raise TWSError(f"Could not reach the app: {e}", method=method, path=path) from e

            # An expired or password-invalidated token. Re-login once, then
            # replay — this is expected in normal long-running operation.
            if resp.status_code == 401 and not relogged and not self._token_is_static:
                relogged = True
                self._token = ""
                self._me = None
                await self.login()
                continue

            if resp.status_code in RETRY_STATUSES and idempotent and attempt + 1 < attempts:
                # Honour Retry-After when the app sends one (its agent API
                # rate-limits), otherwise back off exponentially.
                delay = _retry_after(resp) or (2 ** attempt)
                await asyncio.sleep(delay)
                continue

            if resp.status_code >= 400:
                body = _safe_body(resp)
                await self._log(method, path, resp.status_code, started, body)
                raise TWSError(_describe(resp.status_code, body), status=resp.status_code,
                               method=method, path=path, body=body)

            await self._log(method, path, resp.status_code, started, None)
            if not expect_json:
                return resp.content
            if not resp.content:
                return {}
            try:
                return resp.json()
            except ValueError:
                return {"raw": resp.text}

        raise TWSError(f"Gave up after {attempts} attempts: {last_exc}",
                       method=method, path=path)

    async def _log(self, method: str, path: str, status: Optional[int],
                   started: float, error: Any) -> None:
        if self._audit is None:
            return
        try:
            await self._audit({
                "method": method, "path": path, "status": status,
                "ms": round((time.monotonic() - started) * 1000),
                "error": error,
            })
        except Exception:  # an audit sink must never break the call it records
            log.exception("atlas: audit sink raised")

    # ---------- verbs ----------

    async def run_internal_job(self, path: str) -> Any:
        """Drive one of the app's scheduled jobs on demand.

        These endpoints do not accept a user token at all — they are gated
        solely by the X-Cron-Secret header, and return 401 whenever the app's
        INTERNAL_CRON_SECRET is unset (which is also its default state, so an
        unconfigured app makes them permanently unreachable rather than open).
        """
        if not self._cron_secret:
            raise TWSError(
                "No cron secret configured, so Atlas cannot run the app's scheduled "
                "jobs. Set TWS_CRON_SECRET to the app's INTERNAL_CRON_SECRET.",
                status=401, method="POST", path=path)
        await self.start()
        assert self._client is not None
        url = f"{self.api_root}{path}"
        started = time.monotonic()
        resp = await self._client.post(url, headers={"X-Cron-Secret": self._cron_secret})
        if resp.status_code >= 400:
            body = _safe_body(resp)
            await self._log("POST", path, resp.status_code, started, body)
            if resp.status_code == 401:
                raise TWSError(
                    "The app rejected the cron secret. Either TWS_CRON_SECRET does not "
                    "match the app's INTERNAL_CRON_SECRET, or the app has none set — in "
                    "which case these jobs are unreachable by anyone.",
                    status=401, method="POST", path=path, body=body)
            raise TWSError(_describe(resp.status_code, body), status=resp.status_code,
                           method="POST", path=path, body=body)
        await self._log("POST", path, resp.status_code, started, None)
        try:
            return resp.json()
        except ValueError:
            return {"raw": resp.text}

    async def get(self, path: str, **params: Any) -> Any:
        return await self.request("GET", path, params={k: v for k, v in params.items() if v is not None})

    async def post(self, path: str, body: Optional[dict] = None, **kw: Any) -> Any:
        return await self.request("POST", path, json_body=body or {}, **kw)

    async def put(self, path: str, body: Optional[dict] = None, **kw: Any) -> Any:
        return await self.request("PUT", path, json_body=body or {}, **kw)

    async def patch(self, path: str, body: Optional[dict] = None, **kw: Any) -> Any:
        return await self.request("PATCH", path, json_body=body or {}, **kw)

    async def delete(self, path: str, **kw: Any) -> Any:
        return await self.request("DELETE", path, **kw)


def _retry_after(resp: httpx.Response) -> Optional[float]:
    raw = resp.headers.get("Retry-After")
    if not raw:
        return None
    try:
        return max(0.0, min(60.0, float(raw)))
    except ValueError:
        return None


def _safe_body(resp: httpx.Response) -> Any:
    try:
        data = resp.json()
    except ValueError:
        return (resp.text or "")[:500]
    if isinstance(data, dict) and "detail" in data:
        return data["detail"]
    return data


def _describe(status: int, body: Any) -> str:
    if status == 403:
        return f"Refused: {body}"
    if status == 404:
        return f"Not found: {body}"
    if status == 429:
        return f"Rate limited by the app: {body}"
    return f"HTTP {status}: {body}"
