from __future__ import annotations

from http.cookies import SimpleCookie
from pathlib import Path
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import browser_cookie3
import requests


ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = ROOT / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)
API_PACE_FILE = STATE_DIR / "wq_api_pace.state"
API_LOCK_FILE = STATE_DIR / "wq_api.lock"


class BrainAuthError(RuntimeError):
    pass


class BrainRateLimit(RuntimeError):
    def __init__(self, message: str, retry_after: Optional[int] = None):
        super().__init__(message)
        self.retry_after = retry_after


@dataclass(slots=True)
class BrainClient:
    base_url: str
    browser: str = "chrome"
    cookie_domain: str = ".worldquantbrain.com"
    request_timeout_seconds: int = 30
    user_agent: str = "Mozilla/5.0"
    cookie_file: Optional[str] = None
    session: requests.Session = field(init=False)

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": self.user_agent,
            "Accept": "application/json, text/plain, */*",
            "Content-Type": "application/json",
        })
        if self.cookie_file:
            self.load_cookie_file(self.cookie_file)
        else:
            self.load_browser_cookies()

    def load_cookie_file(self, cookie_file: str | Path) -> None:
        raw_cookie = Path(cookie_file).expanduser().read_text(encoding="utf-8").strip()
        parsed = SimpleCookie()
        parsed.load(raw_cookie)
        for morsel in parsed.values():
            self.session.cookies.set(morsel.key, morsel.value, domain=self.cookie_domain.lstrip("."), path="/")

    def load_browser_cookies(self) -> None:
        loader = getattr(browser_cookie3, self.browser, None)
        if loader is None:
            raise ValueError(f"Unsupported browser '{self.browser}'. Try chrome, brave, firefox, edge, chromium, safari.")
        cookiejar = loader(domain_name=self.cookie_domain)
        self.session.cookies.update(cookiejar)

    def _request(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        self._pace_request(method)
        url = path if path.startswith("http") else f"{self.base_url}{path}"
        response = self.session.request(method, url, timeout=self.request_timeout_seconds, **kwargs)
        if response.status_code in (401, 403):
            raise BrainAuthError("WorldQuant Brain session is unauthorized/expired. Please log in in the browser, then retry.")
        if response.status_code == 429:
            retry_after_header = response.headers.get("Retry-After")
            retry_after = int(retry_after_header) if retry_after_header and retry_after_header.isdigit() else None
            raise BrainRateLimit("WorldQuant Brain rate limit hit.", retry_after=retry_after)
        response.raise_for_status()
        return response

    def _pace_request(self, method: str) -> None:
        # A tiny cross-process throttle reduces the unnatural burst pattern that
        # was causing auth/rate-limit churn when launchd jobs overlapped.
        base_gap = float(os.getenv("WQ_API_MIN_GAP_SECONDS", "2.0"))
        write_gap = float(os.getenv("WQ_API_WRITE_GAP_SECONDS", "8.0"))
        min_gap = write_gap if method.upper() in {"POST", "PATCH", "PUT", "DELETE"} else base_gap
        deadline = time.monotonic() + 30
        lock_fd = None
        while lock_fd is None:
            try:
                lock_fd = os.open(API_LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                os.write(lock_fd, str(os.getpid()).encode("utf-8"))
            except FileExistsError:
                try:
                    if time.time() - API_LOCK_FILE.stat().st_mtime > 60:
                        API_LOCK_FILE.unlink(missing_ok=True)
                        continue
                except Exception:
                    pass
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.2)
        try:
            last = 0.0
            try:
                last = float(API_PACE_FILE.read_text(encoding="utf-8").strip() or "0")
            except Exception:
                pass
            wait = min_gap - (time.time() - last)
            if wait > 0:
                time.sleep(wait)
            API_PACE_FILE.write_text(str(time.time()), encoding="utf-8")
        finally:
            if lock_fd is not None:
                try:
                    os.close(lock_fd)
                except Exception:
                    pass
                try:
                    API_LOCK_FILE.unlink(missing_ok=True)
                except Exception:
                    pass

    def whoami(self) -> dict[str, Any]:
        # Endpoint names occasionally change; this is a light auth probe.
        response = self._request("GET", "/authentication")
        return _json_or_empty(response)

    def create_simulation(self, payload: dict[str, Any]) -> str:
        response = self._request("POST", "/simulations", json=payload)
        location = response.headers.get("Location") or response.headers.get("location")
        if location:
            return location.rstrip("/").split("/")[-1]
        data = _json_or_empty(response)
        simulation_id = data.get("id") or data.get("simulationId")
        if not simulation_id:
            raise RuntimeError(f"Simulation created but no id/location returned: {data or response.text[:500]}")
        return str(simulation_id)

    def get_simulation(self, simulation_id: str) -> dict[str, Any]:
        response = self._request("GET", f"/simulations/{simulation_id}")
        return _json_or_empty(response)

    def get_alpha(self, alpha_id: str) -> dict[str, Any]:
        response = self._request("GET", f"/alphas/{alpha_id}")
        return _json_or_empty(response)

    def submit_alpha(self, alpha_id: str) -> dict[str, Any]:
        # WorldQuant Brain final submit endpoint expects PATCH with a list of
        # alpha objects. A plain PUT is advertised in OPTIONS but returns 405.
        response = self._request("PATCH", f"/alphas/{alpha_id}/submit", json=[{"id": alpha_id}])
        return _json_or_empty(response)

    def wait_for_simulation(self, simulation_id: str, poll_interval_seconds: int = 15, max_poll_seconds: int | None = None) -> dict[str, Any]:
        start = time.monotonic()
        last_data: dict[str, Any] = {}
        while True:
            data = self.get_simulation(simulation_id)
            last_data = data
            status = str(data.get("status") or data.get("state") or "").upper()
            if status in {"COMPLETE", "COMPLETED", "DONE", "ERROR", "FAILED", "FAIL"}:
                return data
            if max_poll_seconds is not None and time.monotonic() - start >= max_poll_seconds:
                return {**last_data, "status": "TIMEOUT", "message": f"poll timeout after {max_poll_seconds}s"}
            time.sleep(poll_interval_seconds)


def _json_or_empty(response: requests.Response) -> dict[str, Any]:
    if not response.content:
        return {}
    try:
        data = response.json()
    except ValueError:
        return {"text": response.text}
    return data if isinstance(data, dict) else {"data": data}
