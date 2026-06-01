#!/usr/bin/env python3
"""Build per-run index from CaMeL workspace logs.

Run from the repo root or anywhere; paths are derived relative to this file.
Writes /tmp/camel_index_v2.jsonl and thesis_analysis/index.csv.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

_HERE = Path(__file__).parent          # thesis_analysis/
_REPO = _HERE.parent                   # repo root

LOGS = _REPO / "logs"
OUT_JSONL = Path("/tmp/camel_index_v2.jsonl")
OUT_CSV = _HERE / "index.csv"

CONFIG_DIRS = [
    "claude-sonnet-4-20250514+camel",
    "claude-sonnet-4-20250514+camel+secpol",
]


def cfg_label(path: str) -> str:
    if "+secpol" in path:
        return "camel+secpol"
    if "important_instructions" in path:
        return "camel_under_attack"
    return "camel_baseline"


def get_text(msg) -> str:
    """Concatenate all string content of a message safely."""
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


# Patterns we look for to flag root-cause signals (used later for tagging).
INTERPRETER_PATTERNS = {
    "break": re.compile(r"Break statements are not supported"),
    "continue": re.compile(r"Continue statements are not supported"),
    "lambda": re.compile(r"[Ll]ambda.*not supported|Lambdas are not supported"),
    "try_except": re.compile(r"[Tt]ry.*not supported|Try-Except.*not supported|TryExcept"),
    "import": re.compile(r"[Ii]mport.*not supported|Imports are not supported"),
    "fstring": re.compile(r"[Ff]-string.*not supported|JoinedStr"),
    "with": re.compile(r"[Ww]ith.*not supported|With statement.*not supported"),
    "starred": re.compile(r"Starred.*not supported|Unpacking.*not supported"),
    "method_call": re.compile(r"\.append\(|\.extend\(|\.pop\(|\.sort\(|\.reverse\(|\.update\("),
    "slicing": re.compile(r"[Ss]licing.*not supported|Slice.*not supported"),
    "global": re.compile(r"[Gg]lobal.*not supported"),
    "nonlocal": re.compile(r"[Nn]onlocal.*not supported"),
    "augassign": re.compile(r"AugAssign|Augmented assignment"),
    "comprehension_complex": re.compile(r"comprehension.*not supported"),
    "yield": re.compile(r"[Yy]ield.*not supported"),
    "assert": re.compile(r"[Aa]ssert.*not supported"),
    "delete": re.compile(r"[Dd]elete.*not supported"),
    "class_def": re.compile(r"ClassDef.*not supported|class definition.*not supported"),
    "decorator": re.compile(r"[Dd]ecorator.*not supported"),
    "match": re.compile(r"[Mm]atch.*not supported"),
}

NAMESPACE_PATTERNS = {
    "redef_class": re.compile(
        r"already defined|cannot be redefined|TypeError.*model|"
        r"PydanticUserError|class .* is already"
    ),
}

QLLM_PATTERNS = {
    "not_enough": re.compile(r"NotEnoughInformationError"),
    "validation": re.compile(r"ValidationError"),
}

SECPOL_PATTERNS = {
    "denied": re.compile(r"SecurityPolicyDeniedError"),
}


def scan_signals(messages):
    """Run all regex flags across the entire transcript and return a flat dict."""
    flags = {}
    full_text_chunks = []
    for m in messages:
        t = get_text(m)
        if t:
            full_text_chunks.append(t)
    full = "\n".join(full_text_chunks)
    for k, p in INTERPRETER_PATTERNS.items():
        flags[f"sig_interp_{k}"] = bool(p.search(full))
    for k, p in NAMESPACE_PATTERNS.items():
        flags[f"sig_ns_{k}"] = bool(p.search(full))
    for k, p in QLLM_PATTERNS.items():
        flags[f"sig_qllm_{k}"] = bool(p.search(full))
    for k, p in SECPOL_PATTERNS.items():
        flags[f"sig_secpol_{k}"] = bool(p.search(full))
    return flags, len(full)


def index_one(path: Path) -> dict:
    with path.open() as f:
        data = json.load(f)
    msgs = data.get("messages", [])
    code_iters = 0
    n_python_blocks = 0
    n_assistant_msgs = 0
    n_tool_msgs = 0
    assist_chars = 0
    user_chars = 0
    tool_chars = 0
    total_tool_calls = 0
    tool_errors = 0
    qai_calls = 0
    for m in msgs:
        role = m.get("role")
        text = get_text(m)
        if role == "assistant":
            n_assistant_msgs += 1
            assist_chars += len(text)
            if "```python" in text:
                code_iters += 1
                n_python_blocks += text.count("```python")
            if "query_ai_assistant" in text:
                qai_calls += text.count("query_ai_assistant(")
            tcs = m.get("tool_calls") or []
            total_tool_calls += len(tcs)
        elif role == "user":
            user_chars += len(text)
        elif role == "tool":
            n_tool_msgs += 1
            tool_chars += len(text)
            if m.get("error"):
                tool_errors += 1
    flags, full_chars = scan_signals(msgs)
    rec = {
        "file": str(path),
        "config": cfg_label(str(path)),
        "user_task": data.get("user_task_id"),
        "injection_task": data.get("injection_task_id"),
        "attack_type": data.get("attack_type"),
        "utility": data.get("utility"),
        "security": data.get("security"),
        "duration": data.get("duration"),
        "error": data.get("error"),
        "n_messages": len(msgs),
        "n_assistant_msgs": n_assistant_msgs,
        "n_tool_msgs": n_tool_msgs,
        "code_iters": code_iters,
        "total_tool_calls": total_tool_calls,
        "tool_errors": tool_errors,
        "qai_calls": qai_calls,
        "assist_chars": assist_chars,
        "user_chars": user_chars,
        "tool_chars": tool_chars,
        "approx_tokens": (assist_chars + user_chars + tool_chars) // 4,
        **flags,
    }
    return rec


def main():
    rows = []
    for cfg in CONFIG_DIRS:
        root = LOGS / cfg / "workspace"
        files = sorted(root.rglob("*.json"))
        print(f"{cfg}: {len(files)} files")
        for fp in files:
            rows.append(index_one(fp))
    with OUT_JSONL.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    cols = list(rows[0].keys())
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {len(rows)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
