from __future__ import annotations

import re
from typing import Any, Callable


IDENT_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")


def split_top_level_args(arg_text: str) -> list[str]:
    args: list[str] = []
    start = 0
    depth = 0
    quote: str | None = None
    for i, ch in enumerate(arg_text):
        if quote:
            if ch == quote:
                quote = None
            continue
        if ch in {"'", '"'}:
            quote = ch
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif ch == "," and depth == 0:
            args.append(arg_text[start:i].strip())
            start = i + 1
    tail = arg_text[start:].strip()
    if tail or arg_text:
        args.append(tail)
    return args


def _parse_function_at(expr: str, name: str, start: int) -> dict[str, Any] | None:
    end_name = start + len(name)
    if start > 0 and expr[start - 1] in IDENT_CHARS:
        return None
    if end_name < len(expr) and expr[end_name] in IDENT_CHARS:
        return None
    open_idx = end_name
    while open_idx < len(expr) and expr[open_idx].isspace():
        open_idx += 1
    if open_idx >= len(expr) or expr[open_idx] != "(":
        return None

    depth = 0
    quote: str | None = None
    close_idx = None
    for i in range(open_idx, len(expr)):
        ch = expr[i]
        if quote:
            if ch == quote:
                quote = None
            continue
        if ch in {"'", '"'}:
            quote = ch
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                close_idx = i
                break
    if close_idx is None:
        return None

    arg_text = expr[open_idx + 1 : close_idx]
    return {
        "function": name,
        "start": start,
        "end": close_idx + 1,
        "open": open_idx,
        "close": close_idx,
        "full_match": expr[start : close_idx + 1],
        "args_text": arg_text,
        "args": split_top_level_args(arg_text),
    }


def find_function_calls(expr: str, name: str) -> list[dict[str, Any]]:
    """Find calls, including nested calls, while respecting parentheses."""
    out: list[dict[str, Any]] = []

    def scan(text: str, base: int = 0) -> None:
        i = 0
        while True:
            start = text.find(name, i)
            if start < 0:
                return
            call = _parse_function_at(text, name, start)
            if not call:
                i = start + len(name)
                continue
            shifted = dict(call)
            shifted["start"] = base + int(call["start"])
            shifted["end"] = base + int(call["end"])
            shifted["open"] = base + int(call["open"])
            shifted["close"] = base + int(call["close"])
            out.append(shifted)
            for arg in call["args"]:
                scan(arg, 0)
            i = int(call["end"])

    scan(expr or "")
    return out


def ts_corr_calls(expr: str) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for call in find_function_calls(expr or "", "ts_corr"):
        args = list(call.get("args") or [])
        if len(args) < 3:
            continue
        item = dict(call)
        item.update({
            "field_a": args[0].strip(),
            "field_b": args[1].strip(),
            "window": args[2].strip(),
        })
        calls.append(item)
    return calls


def expression_has_token(expr: str, tokens: set[str]) -> bool:
    lower = expr.lower()
    return any(re.search(rf"\b{re.escape(token.lower())}\b", lower) for token in tokens)


def replace_ts_corr_second_args(
    expr: str,
    should_replace: Callable[[str], bool],
    new_target: str,
) -> str:
    """Replace second arguments in ts_corr calls, including nested calls."""

    def transform(text: str) -> str:
        out: list[str] = []
        i = 0
        while True:
            start = text.find("ts_corr", i)
            if start < 0:
                out.append(text[i:])
                break
            call = _parse_function_at(text, "ts_corr", start)
            if not call:
                out.append(text[i : start + len("ts_corr")])
                i = start + len("ts_corr")
                continue
            out.append(text[i:start])
            args = [transform(arg) for arg in call.get("args", [])]
            if len(args) >= 2 and should_replace(args[1].strip()):
                args[1] = new_target
            replacement = "ts_corr(" + ", ".join(args) + ")"
            out.append(replacement)
            i = int(call["end"])
        return "".join(out)

    return transform(expr or "")
