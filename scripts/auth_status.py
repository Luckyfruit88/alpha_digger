#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alpha_factory.backtester import load_config
from alpha_factory.brain_client import BrainClient


def main() -> int:
    parser = argparse.ArgumentParser(description="Strict WorldQuant auth status probe")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        cfg = load_config(ROOT / "config.yaml")
        info = BrainClient(**cfg["brain"]).whoami()
        user = ((info or {}).get("user") or {}).get("id") if isinstance(info, dict) else None
        token = (info or {}).get("token") if isinstance(info, dict) else None
        expiry = token.get("expiry") if isinstance(token, dict) else None
        ok = bool(user or token)
        payload = {"ok": ok, "user": user, "expiry_seconds": expiry, "raw_has_user_or_token": ok}
        if args.json:
            print(json.dumps(payload, ensure_ascii=False))
        else:
            print("ok" if ok else "expired", payload)
        return 0 if ok else 2
    except Exception as exc:
        payload = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        if args.json:
            print(json.dumps(payload, ensure_ascii=False))
        else:
            print("expired", payload)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
