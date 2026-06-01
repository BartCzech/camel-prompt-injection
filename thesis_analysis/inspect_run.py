#!/usr/bin/env python3
"""Pretty-print a CaMeL log run for manual root-cause classification.

Usage:
    python3 inspect_run.py <path-to-json>
    python3 inspect_run.py --task camel_baseline:user_task_11
    python3 inspect_run.py --task camel_under_attack:user_task_4:injection_task_3
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_HERE = Path(__file__).parent          # thesis_analysis/
_REPO = _HERE.parent                   # repo root

LOGS = _REPO / "logs"


def cfg_dir(cfg: str) -> str:
    return {
        "camel_baseline": "claude-sonnet-4-20250514+camel",
        "camel_under_attack": "claude-sonnet-4-20250514+camel",
        "camel+secpol": "claude-sonnet-4-20250514+camel+secpol",
    }[cfg]


def resolve(spec: str) -> Path:
    parts = spec.split(":")
    cfg = parts[0]
    cdir = cfg_dir(cfg)
    if cfg == "camel_baseline":
        return LOGS / cdir / "workspace" / parts[1] / "none" / "none.json"
    if cfg == "camel+secpol":
        return LOGS / cdir / "workspace" / parts[1] / "none" / "none.json"
    if cfg == "camel_under_attack":
        return LOGS / cdir / "workspace" / parts[1] / "important_instructions" / f"{parts[2]}.json"
    raise ValueError(spec)


def get_text(msg) -> str:
    parts = []
    content = msg.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    for c in content:
        if isinstance(c, dict):
            v = c.get("content")
            if isinstance(v, str):
                parts.append(v)
        elif isinstance(c, str):
            parts.append(c)
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", help="config:user_task[:injection_task]")
    ap.add_argument("path", nargs="?")
    ap.add_argument("--full", action="store_true", help="dump full message text")
    args = ap.parse_args()

    if args.task:
        path = resolve(args.task)
    elif args.path:
        path = Path(args.path)
    else:
        ap.error("need --task or path")

    with path.open() as f:
        data = json.load(f)

    print(f"=== {path.name} ===")
    print(f"user_task: {data.get('user_task_id')}")
    print(f"injection_task: {data.get('injection_task_id')}")
    print(f"utility: {data.get('utility')}   security: {data.get('security')}")
    print(f"duration: {data.get('duration'):.2f}s   error: {data.get('error')!r}")
    print(f"n_messages: {len(data['messages'])}")
    print()

    msgs = data["messages"]
    user_q = ""
    for m in msgs:
        if m.get("role") == "user":
            user_q = get_text(m)
            break
    print("USER PROMPT:")
    print("  " + user_q.strip().replace("\n", "\n  "))
    print()

    iter_idx = 0
    for i, m in enumerate(msgs):
        role = m.get("role")
        text = get_text(m)
        if role == "assistant":
            if "```python" in text:
                iter_idx += 1
                pre = text.split("```python")[0].strip()
                code = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
                code_body = code.group(1) if code else ""
                post = text.split("```")[-1].strip() if text.count("```") >= 2 else ""
                print(f"--- ASSISTANT iter {iter_idx} (msg {i}) ---")
                if pre:
                    print(f"[narration] {pre[:300]}{'...' if len(pre)>300 else ''}")
                print("[code]")
                for ln in code_body.splitlines():
                    print("  " + ln)
                if post:
                    print(f"[trailing] {post[:200]}")
                print()
            else:
                # final answer or text-only
                snippet = text.strip()[:600]
                print(f"--- ASSISTANT (msg {i}, no python) ---")
                print(f"  {snippet}")
                print()
        elif role == "user" and i > 0:
            # interpreter feedback / error
            if "Running the code gave the following error" in text or "SecurityPolicyDenied" in text:
                lines = text.strip().splitlines()
                # Print the error part only
                print(f"--- USER feedback (msg {i}) ---")
                for ln in lines[:25]:
                    print(f"  {ln}")
                if len(lines) > 25:
                    print(f"  ... ({len(lines)-25} more lines)")
                print()
        elif role == "tool":
            tc = m.get("tool_call") or {}
            fn = tc.get("function", "?")
            err = m.get("error")
            content = text
            preview = content.replace("\n", " ")[:140]
            err_str = f" ERR={err!r}" if err else ""
            print(f"  [tool] {fn}({tc.get('args', {})}) -> {preview!r}{err_str}")

    if data.get("messages"):
        last = data["messages"][-1]
        print()
        print("FINAL ASSISTANT:")
        print("  " + get_text(last).strip().replace("\n", "\n  ")[:1000])


if __name__ == "__main__":
    main()
