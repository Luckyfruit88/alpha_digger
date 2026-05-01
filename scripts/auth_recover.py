#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from http.cookies import SimpleCookie
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alpha_factory.backtester import load_config
from alpha_factory.brain_client import BrainClient

STATE_DIR = ROOT / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_PATH = STATE_DIR / "auth_recovery_state.json"
AUTONOMY_ENV = ROOT / "secrets" / "autonomy.env"
DEFAULT_LOGIN_URL = "https://platform.worldquantbrain.com/sign-in"
DEFAULT_CREDENTIALS_FILE = "secrets/worldquant_credentials.json"
UTC = timezone.utc


def now_utc() -> datetime:
    return datetime.now(UTC)


def load_env_file(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    if not path.exists():
        return data
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip()
    return data


def load_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def load_state() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return {"attempts": []}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"attempts": []}


def save_state(state: dict[str, Any]) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def run(cmd: list[str], *, timeout: int = 60, input_text: str | None = None, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, input=input_text, capture_output=True, text=True, timeout=timeout, check=check)


def probe_auth() -> tuple[bool, str]:
    try:
        cfg = load_config(ROOT / "config.yaml")
        client = BrainClient(**cfg["brain"])
        info = client.whoami()
        user = ((info or {}).get("user") or {}).get("id")
        token = (info or {}).get("token") if isinstance(info, dict) else None
        if user or token:
            return True, f"auth ok user={user or 'unknown'}"
        return False, f"auth probe returned no user/token: {info or {}}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def paste_text_secure(text: str) -> None:
    """Paste text without placing secrets in command-line arguments.

    Uses the system clipboard via stdin, then restores the previous clipboard
    best-effort. The secret may briefly exist in the user's clipboard, but it is
    not printed or exposed in ps/tool argv.
    """
    old_clip = run(["pbpaste"], timeout=10).stdout
    try:
        run(["pbcopy"], input_text=text, timeout=10)
        if shutil_which("peekaboo"):
            run(["peekaboo", "hotkey", "--keys", "cmd,v", "--app", "Safari"], timeout=20)
        else:
            run(["osascript", "-e", 'tell application "System Events" to keystroke "v" using {command down}'], timeout=20)
    finally:
        run(["pbcopy"], input_text=old_clip, timeout=10)


def cooldown_allows(state: dict[str, Any], env: dict[str, str]) -> tuple[bool, str]:
    attempts = [a for a in state.get("attempts", []) if isinstance(a, dict) and a.get("ts")]
    cutoff = now_utc() - timedelta(hours=24)
    recent: list[dict[str, Any]] = []
    for attempt in attempts:
        try:
            ts = datetime.fromisoformat(str(attempt["ts"]))
            if ts >= cutoff:
                recent.append(attempt)
        except Exception:
            continue
    state["attempts"] = recent
    max_day = int(env.get("WQ_RECOVERY_MAX_ATTEMPTS_PER_DAY") or 8)
    # max_day <= 0 means no fixed daily cap. Keep cooldown/backoff in place so
    # transient auth/network failures do not become tight login loops.
    if max_day > 0 and len(recent) >= max_day:
        return False, f"daily recovery attempt cap reached ({len(recent)}/{max_day})"
    if recent:
        last = datetime.fromisoformat(str(recent[-1]["ts"]))
        cooldown = int(env.get("WQ_RECOVERY_COOLDOWN_MINUTES") or 30)
        remaining = timedelta(minutes=cooldown) - (now_utc() - last)
        if remaining.total_seconds() > 0:
            return False, f"cooldown active for {int(remaining.total_seconds())}s"
    return True, "ok"


def record_attempt(state: dict[str, Any], status: str, detail: str) -> None:
    state.setdefault("attempts", []).append({"ts": now_utc().isoformat(), "status": status, "detail": detail[:500]})
    state["last_status"] = {"ts": now_utc().isoformat(), "status": status, "detail": detail[:500]}
    save_state(state)


def find_keychain_account(env: dict[str, str]) -> str | None:
    account = env.get("WQ_KEYCHAIN_ACCOUNT") or ""
    return account or None


def read_keychain_password(env: dict[str, str]) -> tuple[str | None, str | None]:
    """Return (account, password) if explicitly configured; never logs password."""
    service = env.get("WQ_KEYCHAIN_SERVICE") or "WorldQuantBrain"
    account = find_keychain_account(env)
    if not account:
        return None, None
    try:
        result = run(["security", "find-generic-password", "-s", service, "-a", account, "-w"], timeout=30)
    except Exception:
        return account, None
    password = result.stdout.strip() if result.returncode == 0 else None
    return account, password


def read_json_credentials(env: dict[str, str]) -> tuple[str | None, str | None, Path]:
    """Return (account, password, path) from a private local JSON file.

    Supported keys: email/password, username/password, account/password.
    Placeholder/template values are treated as unset so the script can be
    safely installed before the user fills in credentials.
    """
    rel = env.get("WQ_CREDENTIALS_FILE") or DEFAULT_CREDENTIALS_FILE
    path = (ROOT / rel).expanduser() if not rel.startswith("/") else Path(rel).expanduser()
    payload = load_json_file(path)
    account = str(payload.get("email") or payload.get("username") or payload.get("account") or "").strip()
    password = str(payload.get("password") or "").strip()
    placeholders = {"", "YOUR_WORLDQUANT_EMAIL", "YOUR_WORLDQUANT_PASSWORD", "change-me", "changeme"}
    if account in placeholders or password in placeholders:
        return None, None, path
    return account, password, path


def write_cookie_file_from_session(session: requests.Session, cookie_file: Path) -> str:
    parts: list[str] = []
    for cookie in session.cookies:
        if cookie.name and cookie.value and ("worldquant" in (cookie.domain or "") or cookie.name in {"t", "cookieyes-consent"}):
            parts.append(f"{cookie.name}={cookie.value}")
    if not parts:
        raise RuntimeError("login succeeded but no WorldQuant cookies were returned")
    cookie_file.parent.mkdir(parents=True, exist_ok=True)
    cookie_file.write_text("; ".join(parts), encoding="utf-8")
    try:
        os.chmod(cookie_file, 0o600)
    except Exception:
        pass
    return f"wrote {len(parts)} cookies to {cookie_file.relative_to(ROOT)}"


def api_login_with_json_credentials(env: dict[str, str]) -> str | None:
    """Refresh the Brain API cookie via the supported authentication endpoint.

    The WorldQuant Brain API accepts Basic Auth on POST /authentication and
    returns session cookies for valid credentials. This path avoids UI
    automation and does not log the credential values.
    """
    account, password, credentials_path = read_json_credentials(env)
    if not account or not password:
        return None
    cfg = load_config(ROOT / "config.yaml")
    brain = cfg["brain"]
    base_url = str(brain["base_url"]).rstrip("/")
    timeout = int(brain.get("request_timeout_seconds") or 30)
    session = requests.Session()
    session.headers.update({
        "User-Agent": str(brain.get("user_agent") or "Mozilla/5.0"),
        "Accept": "application/json, text/plain, */*",
    })
    response = session.post(f"{base_url}/authentication", auth=(account, password), timeout=timeout)
    if response.status_code in {401, 403}:
        raise RuntimeError(f"JSON credential login rejected for account {account}: HTTP {response.status_code}")
    response.raise_for_status()
    cookie_path = ROOT / (env.get("WQ_COOKIE_FILE") or "secrets/worldquant.cookie")
    cookie_detail = write_cookie_file_from_session(session, cookie_path)
    try:
        os.chmod(credentials_path, 0o600)
    except Exception:
        pass
    return f"JSON credential login ok for account {account}; {cookie_detail}"


def safari_open(url: str) -> None:
    script = f'''
    tell application "Safari"
      activate
      if (count of windows) = 0 then make new document
      set URL of front document to "{url}"
    end tell
    '''
    run(["osascript", "-e", script], timeout=30)


def safari_current_url() -> str:
    script = 'tell application "Safari" to if (count of windows) > 0 then return URL of front document'
    result = run(["osascript", "-e", script], timeout=30)
    return result.stdout.strip()


def safari_page_text() -> str:
    script = '''
    tell application "Safari"
      if (count of windows) = 0 then return ""
      try
        return text of front document
      on error
        return ""
      end try
    end tell
    '''
    result = run(["osascript", "-e", script], timeout=30)
    return result.stdout


def safari_autofill_login(env: dict[str, str]) -> str:
    """Open Safari and attempt a safe login using existing Keychain/Safari autofill.

    If configured Keychain account/password are present, paste them into the page.
    Otherwise, ask Safari/iCloud Passwords to autofill via keyboard shortcut. If
    MFA/CAPTCHA appears, stop and report rather than trying to bypass it.
    """
    login_url = env.get("WQ_LOGIN_URL") or DEFAULT_LOGIN_URL
    timeout_seconds = int(env.get("WQ_LOGIN_TIMEOUT_SECONDS") or 180)
    post_login_wait = int(env.get("WQ_POST_LOGIN_WAIT_SECONDS") or 8)
    safari_open(login_url)
    time.sleep(5)

    account, password = read_keychain_password(env)
    if account and password:
        if shutil_which("peekaboo"):
            run(["peekaboo", "hotkey", "--keys", "cmd,l"], timeout=20)
            run(["peekaboo", "press", "tab"], timeout=20)
            paste_text_secure(account)
            run(["peekaboo", "press", "tab"], timeout=20)
            paste_text_secure(password)
            run(["peekaboo", "press", "return"], timeout=20)
        else:
            paste_text_secure(account)
            run(["osascript", "-e", 'tell application "System Events" to key code 48'], timeout=20)
            paste_text_secure(password)
            run(["osascript", "-e", 'tell application "System Events" to key code 36'], timeout=20)
    else:
        # Trigger Safari password autofill if credentials are saved in iCloud
        # Passwords. This may show a system prompt; if it requires biometrics or
        # MFA, the later auth probe will fail and the user is notified.
        if shutil_which("peekaboo"):
            run(["peekaboo", "hotkey", "--keys", "cmd,shift,l"], timeout=20)
            time.sleep(1)
            run(["peekaboo", "press", "return"], timeout=20)
        else:
            run(["osascript", "-e", 'tell application "System Events" to keystroke "l" using {command down, shift down}'], timeout=20)
            time.sleep(1)
            run(["osascript", "-e", 'tell application "System Events" to key code 36'], timeout=20)

    deadline = time.time() + timeout_seconds
    last_reason = "waiting for login"
    while time.time() < deadline:
        ok, detail = probe_auth()
        if ok:
            time.sleep(post_login_wait)
            return f"login recovered: {detail}"
        page_text = safari_page_text().lower()
        current_url = safari_current_url().lower()
        if any(token in page_text or token in current_url for token in ["captcha", "verification", "verify", "multi-factor", "mfa", "otp", "code"]):
            raise RuntimeError("manual verification required (MFA/CAPTCHA/OTP detected)")
        last_reason = detail
        time.sleep(5)
    raise RuntimeError(f"login recovery timed out: {last_reason}")


def shutil_which(name: str) -> bool:
    try:
        return run(["/usr/bin/which", name], timeout=10).returncode == 0
    except Exception:
        return False


def export_safari_cookie(cookie_file: Path) -> str:
    import browser_cookie3

    jar = browser_cookie3.safari(domain_name=".worldquantbrain.com")
    parts = []
    for cookie in jar:
        if cookie.name in {"t", "cookieyes-consent"} or "worldquant" in (cookie.domain or ""):
            parts.append(f"{cookie.name}={cookie.value}")
    if not parts:
        raise RuntimeError("no WorldQuant cookies found in Safari after login")
    cookie_file.parent.mkdir(parents=True, exist_ok=True)
    cookie_file.write_text("; ".join(parts), encoding="utf-8")
    try:
        os.chmod(cookie_file, 0o600)
    except Exception:
        pass
    return f"wrote {len(parts)} cookies to {cookie_file.relative_to(ROOT)}"


def main() -> None:
    env = load_env_file(AUTONOMY_ENV)
    state = load_state()
    ok, detail = probe_auth()
    if ok:
        print(json.dumps({"status": "ok", "detail": detail}, ensure_ascii=False))
        return

    allowed, reason = cooldown_allows(state, env)
    if not allowed:
        print(json.dumps({"status": "skipped", "detail": reason, "auth_error": detail}, ensure_ascii=False))
        return

    record_attempt(state, "started", detail)
    try:
        login_detail = api_login_with_json_credentials(env)
        if login_detail is None:
            login_detail = safari_autofill_login(env)
            cookie_path = ROOT / (env.get("WQ_COOKIE_FILE") or "secrets/worldquant.cookie")
            cookie_detail = export_safari_cookie(cookie_path)
            login_detail = f"{login_detail}; {cookie_detail}"
        ok, final_detail = probe_auth()
        if not ok:
            raise RuntimeError(f"cookie exported but auth still failed: {final_detail}")
        record_attempt(state, "success", f"{login_detail}; {final_detail}")
        print(json.dumps({"status": "recovered", "detail": final_detail}, ensure_ascii=False))
    except Exception as exc:
        record_attempt(state, "failed", str(exc))
        print(json.dumps({"status": "failed", "detail": str(exc)}, ensure_ascii=False))
        raise SystemExit(2)


if __name__ == "__main__":
    main()
