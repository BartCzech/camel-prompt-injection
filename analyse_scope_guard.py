# Copyright 2026 Bartłomiej Czech
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compare scope-guarded branching results vs. CaMeL baseline.

Usage (after the run completes):
    uv run python analyse_scope_guard.py

Reads from:
    logs/claude-sonnet-4-20250514+camel/workspace/          (baseline)
    logs/claude-sonnet-4-20250514+camel+scope-guard/workspace/  (scope-guard)

Writes:
    results/scope_guard_comparison.md
"""

from __future__ import annotations

import json
from pathlib import Path

BASELINE_DIR = Path("logs/claude-sonnet-4-20250514+camel/workspace")
SCOPE_GUARD_DIR = Path("logs/claude-sonnet-4-20250514+camel+scope-guard/workspace")
OUT_FILE = Path("results/scope_guard_comparison.md")


def load_utility(log_dir: Path) -> dict[str, bool]:
    """Return {task_id: utility_bool} for all user tasks in log_dir."""
    results: dict[str, bool] = {}
    for task_dir in sorted(log_dir.iterdir()):
        if not task_dir.name.startswith("user_task_"):
            continue
        task_id = task_dir.name
        # benign run lives under none/none.json
        json_path = task_dir / "none" / "none.json"
        if not json_path.exists():
            continue
        data = json.loads(json_path.read_text())
        results[task_id] = bool(data.get("utility", False))
    return results


def main() -> None:
    if not BASELINE_DIR.exists():
        print(f"ERROR: baseline dir not found: {BASELINE_DIR}")
        return
    if not SCOPE_GUARD_DIR.exists():
        print(f"ERROR: scope-guard dir not found: {SCOPE_GUARD_DIR}")
        return

    baseline = load_utility(BASELINE_DIR)
    scope_guard = load_utility(SCOPE_GUARD_DIR)

    all_tasks = sorted(set(baseline) | set(scope_guard))

    pass_to_fail: list[str] = []
    fail_to_pass: list[str] = []
    both_pass: list[str] = []
    both_fail: list[str] = []
    only_baseline: list[str] = []
    only_sg: list[str] = []

    for t in all_tasks:
        b = baseline.get(t)
        s = scope_guard.get(t)
        if b is None:
            only_sg.append(t)
        elif s is None:
            only_baseline.append(t)
        elif b and s:
            both_pass.append(t)
        elif b and not s:
            pass_to_fail.append(t)
        elif not b and s:
            fail_to_pass.append(t)
        else:
            both_fail.append(t)

    total_b = sum(baseline.values())
    total_sg = sum(v for t, v in scope_guard.items() if t in baseline)
    n = len([t for t in all_tasks if t in baseline and t in scope_guard])

    lines: list[str] = [
        "# Scope-Guard vs Baseline Comparison",
        "",
        f"Tasks compared: {n}/40",
        f"Baseline utility:     {total_b}/{len(baseline)} = {total_b/len(baseline)*100:.1f}%",
        f"Scope-guard utility:  {total_sg}/{n} = {total_sg/n*100:.1f}% (on same {n} tasks)",
        "",
        f"## Changes",
        f"  Pass → Fail (regressions):  {len(pass_to_fail)}",
        f"  Fail → Pass (recoveries):   {len(fail_to_pass)}",
        f"  Both pass (unchanged):      {len(both_pass)}",
        f"  Both fail (unchanged):      {len(both_fail)}",
    ]

    if pass_to_fail:
        lines += ["", "### Regressions (pass → fail)"]
        for t in pass_to_fail:
            lines.append(f"  - {t}")

    if fail_to_pass:
        lines += ["", "### Recoveries (fail → pass)"]
        for t in fail_to_pass:
            lines.append(f"  - {t}")

    if only_baseline:
        lines += ["", f"### Only in baseline (not yet in scope-guard run): {len(only_baseline)}"]
        for t in only_baseline:
            lines.append(f"  - {t}")

    if only_sg:
        lines += ["", f"### Only in scope-guard (not in baseline): {len(only_sg)}"]
        for t in only_sg:
            lines.append(f"  - {t}")

    report = "\n".join(lines) + "\n"
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(report)
    print(report)
    print(f"\nWritten to {OUT_FILE}")


if __name__ == "__main__":
    main()
