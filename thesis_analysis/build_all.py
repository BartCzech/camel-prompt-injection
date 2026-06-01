"""
Build all CSV deliverables for the CaMeL workspace failure-mode analysis:
    - failures_baseline.csv          (9 rows)
    - failures_secpol.csv            (9 rows, regressions only)
    - failures_under_attack.csv      (112 rows)
    - sample_passed_expensive.csv    (38 iter>=4 + ~115 random sample)
    - aggregate.csv                  (cross-tab category x arch/impl/n.a.)
    - per-task index.csv             (already exists; we re-emit a 640-row clean version)

Usage:
    cd thesis_analysis
    python3 build_all.py
"""

from __future__ import annotations
import csv
import json
import random
import sys
from pathlib import Path
from collections import defaultdict
from statistics import mean, median

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from tags import (
    BASELINE_TAGS,
    SECPOL_REGRESSION_TAGS,
    category_arch_or_impl,
    all_categories,
)

INDEX_JSONL = Path("/tmp/camel_index_v2.jsonl")


def load_index() -> list[dict]:
    rows = [json.loads(l) for l in INDEX_JSONL.open()]
    return rows


def is_user_task_row(r: dict) -> bool:
    return "/user_task_" in r["file"]


# ---------------------------------------------------------------------------

PER_TASK_FIELDS = [
    "task_id",
    "config",
    "injection_task",
    "utility",
    "security",
    "code_iters",
    "total_tool_calls",
    "duration_seconds",
    "error_present",
    "applicable_root_causes",
    "primary_cause",
    "primary_cause_arch_impl",
    "classification_confidence",
    "approx_tokens",
    "narrative",
]


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def make_per_task_row(idx_row: dict, tag_entry: dict) -> dict:
    primary = tag_entry["primary"]
    return {
        "task_id": idx_row["user_task"],
        "config": idx_row["config"],
        "injection_task": idx_row.get("injection_task", "") or "",
        "utility": idx_row["utility"],
        "security": idx_row.get("security", ""),
        "code_iters": idx_row["code_iters"],
        "total_tool_calls": idx_row["total_tool_calls"],
        "duration_seconds": round(idx_row["duration"], 2),
        "error_present": bool(idx_row.get("error")),
        "applicable_root_causes": "|".join(tag_entry["tags"]),
        "primary_cause": primary,
        "primary_cause_arch_impl": category_arch_or_impl(primary),
        "classification_confidence": tag_entry["confidence"],
        "approx_tokens": idx_row.get("approx_tokens", 0),
        "narrative": tag_entry["narrative"],
    }


def make_full_failure_row(idx_row: dict, tag_entry: dict, **extras) -> dict:
    base = make_per_task_row(idx_row, tag_entry)
    base["arch_or_impl"] = tag_entry["arch_or_impl"]
    base["user_prompt"] = tag_entry["user_prompt"]
    base["code_excerpt"] = tag_entry["code_excerpt"]
    base.update(extras)
    return base


# ---------------------------------------------------------------------------


def build_failures_baseline(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        if r["config"] != "camel_baseline":
            continue
        if r["utility"]:
            continue
        if not is_user_task_row(r):
            continue
        ut = r["user_task"]
        tag = BASELINE_TAGS[ut]
        out.append(make_full_failure_row(r, tag))
    out.sort(key=lambda x: int(x["task_id"].split("_")[-1]))
    return out


def build_failures_secpol(rows: list[dict]) -> list[dict]:
    out = []
    baseline_pass = {
        r["user_task"]
        for r in rows
        if r["config"] == "camel_baseline" and r["utility"] and is_user_task_row(r)
    }
    for r in rows:
        if r["config"] != "camel+secpol":
            continue
        if r["utility"]:
            continue
        if not is_user_task_row(r):
            continue
        if r["user_task"] not in baseline_pass:
            continue  # only regressions in this CSV
        tag = SECPOL_REGRESSION_TAGS[r["user_task"]]
        out.append(
            make_full_failure_row(r, tag, secpol_subcause=tag.get("secpol_subcause", ""))
        )
    out.sort(key=lambda x: int(x["task_id"].split("_")[-1]))
    return out


def build_failures_under_attack(rows: list[dict]) -> list[dict]:
    """
    Propagate baseline failure tags to every under-attack failure for the
    same user_task. Add an injection_effect column = "no_effect_pre_existing"
    based on the spot-check evidence that all 112 under-attack failures map
    onto user_tasks that already failed in baseline (i.e. zero injection-induced
    new failures).
    """
    baseline_fail_tasks = set(BASELINE_TAGS.keys())
    out = []
    for r in rows:
        if r["config"] != "camel_under_attack":
            continue
        if r["utility"]:
            continue
        if not is_user_task_row(r):
            continue
        ut = r["user_task"]
        if ut in baseline_fail_tasks:
            tag = BASELINE_TAGS[ut]
            inj_effect = "no_effect_pre_existing_baseline_failure"
            inj_conf = "high"
        else:
            # If we ever observed an injection-only failure (a task that passes
            # in baseline but fails under attack) we'd need a different tag.
            tag = {
                "tags": ["?"],
                "primary": "?",
                "confidence": "low",
                "narrative": (
                    "Injection-induced failure with no baseline analogue; "
                    "needs manual classification."
                ),
                "arch_or_impl": "n/a",
                "user_prompt": "(see baseline tag for prompt)",
                "code_excerpt": "(see trace)",
            }
            inj_effect = "injection_induced"
            inj_conf = "low"
        row = make_full_failure_row(
            r,
            tag,
            injection_effect=inj_effect,
            injection_effect_confidence=inj_conf,
        )
        out.append(row)
    out.sort(key=lambda x: (int(x["task_id"].split("_")[-1]), x["injection_task"]))
    return out


# ---------------------------------------------------------------------------
# Stratified sample of passed runs (iter >= 4 fully + 25% random below)


def signal_to_tags(idx_row: dict) -> tuple[list[str], str, str, str]:
    """Heuristic tagging of a passed-but-expensive run from index signals."""
    tags = []
    interp_signals = [
        "sig_interp_break",
        "sig_interp_continue",
        "sig_interp_lambda",
        "sig_interp_try_except",
        "sig_interp_import",
        "sig_interp_fstring",
        "sig_interp_with",
        "sig_interp_starred",
        "sig_interp_method_call",
        "sig_interp_slicing",
        "sig_interp_global",
        "sig_interp_nonlocal",
        "sig_interp_augassign",
        "sig_interp_comprehension_complex",
        "sig_interp_yield",
        "sig_interp_assert",
        "sig_interp_delete",
        "sig_interp_class_def",
        "sig_interp_decorator",
        "sig_interp_match",
    ]
    if any(idx_row.get(s) for s in interp_signals):
        tags.append("A")
    if idx_row.get("sig_qllm_not_enough") or idx_row.get("sig_qllm_validation"):
        tags.append("B")
    if idx_row.get("sig_ns_redef_class"):
        tags.append("F")
    if idx_row.get("sig_secpol_denied"):
        tags.append("C")
    if not tags:
        tags = ["none"]
        primary = "none"
        conf = "high"
        narrative = "Run finished with utility=True and no flagged friction signals."
    else:
        primary = "A" if "A" in tags else tags[0]
        # confidence depends on iter count
        conf = "medium" if idx_row["code_iters"] >= 3 else "low"
        if idx_row["code_iters"] >= 5:
            conf = "high"
        narrative_parts = []
        if "A" in tags:
            interp_kinds = []
            for k, label in [
                ("sig_interp_break", "break"),
                ("sig_interp_continue", "continue"),
                ("sig_interp_lambda", "lambda"),
                ("sig_interp_try_except", "try_except"),
                ("sig_interp_method_call", ".append/.extend/etc"),
                ("sig_interp_slicing", "slicing"),
                ("sig_interp_import", "import"),
                ("sig_interp_fstring", "fstring"),
                ("sig_interp_with", "with"),
                ("sig_interp_starred", "starred"),
                ("sig_interp_global", "global"),
                ("sig_interp_class_def", "class_def"),
                ("sig_interp_decorator", "decorator"),
                ("sig_interp_match", "match"),
                ("sig_interp_yield", "yield"),
                ("sig_interp_assert", "assert"),
                ("sig_interp_delete", "delete"),
                ("sig_interp_comprehension_complex", "complex_comprehension"),
            ]:
                if idx_row.get(k):
                    interp_kinds.append(label)
            narrative_parts.append(
                "Interpreter friction (" + ", ".join(interp_kinds) + ") forced re-iteration"
            )
        if "B" in tags:
            narrative_parts.append("at least one query_ai_assistant raised NotEnoughInformationError")
        if "F" in tags:
            narrative_parts.append("class redefinition rejected at least once")
        if "C" in tags:
            narrative_parts.append("SecurityPolicyDeniedError observed (only secpol replays)")
        narrative = "; ".join(narrative_parts) + "; task ultimately succeeded."
    primary_arch = category_arch_or_impl(primary) if primary != "none" else "n/a"
    return tags, primary, conf, narrative


def build_passed_expensive_sample(rows: list[dict], rng_seed: int = 42) -> list[dict]:
    user_rows = [r for r in rows if is_user_task_row(r)]
    passed = [r for r in user_rows if r["utility"]]
    expensive = [r for r in passed if r["code_iters"] >= 4]
    cheap = [r for r in passed if r["code_iters"] < 4]
    rng = random.Random(rng_seed)
    sample_cheap = rng.sample(cheap, k=len(cheap) // 4)

    out = []
    for r in expensive + sample_cheap:
        tags, primary, conf, narrative = signal_to_tags(r)
        primary_arch = category_arch_or_impl(primary) if primary != "none" else "n/a"
        out.append(
            {
                "task_id": r["user_task"],
                "config": r["config"],
                "injection_task": r.get("injection_task", "") or "",
                "utility": r["utility"],
                "security": r.get("security", ""),
                "code_iters": r["code_iters"],
                "total_tool_calls": r["total_tool_calls"],
                "duration_seconds": round(r["duration"], 2),
                "approx_tokens": r.get("approx_tokens", 0),
                "applicable_root_causes": "|".join(tags),
                "primary_cause": primary,
                "primary_cause_arch_impl": primary_arch,
                "classification_confidence": conf,
                "stratum": "iter_ge_4" if r["code_iters"] >= 4 else "random_25pct_iter_lt_4",
                "narrative": narrative,
            }
        )
    out.sort(
        key=lambda x: (
            -1 if x["stratum"] == "iter_ge_4" else 0,
            x["task_id"],
            x["config"],
            x["injection_task"],
        )
    )
    return out


# ---------------------------------------------------------------------------
# Aggregate cross-tab: rows = categories A..I, cols = (architectural,
# implementation, n/a).  Cells = count + mean iters + mean tokens + mean dur


def build_aggregate(
    failures_baseline, failures_secpol, failures_attack, sample_passed
):
    """
    For each tagged failure, increment a cell for every category that appears
    in the multi-label set (so a row with tags A, B, F counts in three rows).
    The arch/impl bucket is determined per-category, not per-row.
    """
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    src_label = []  # (rows, source_label) tuples for diagnostics
    src_label.append((failures_baseline, "baseline"))
    src_label.append((failures_secpol, "secpol"))
    src_label.append((failures_attack, "under_attack"))
    src_label.append((sample_passed, "sample_passed"))

    for src_rows, src_name in src_label:
        for r in src_rows:
            tag_str = r.get("applicable_root_causes", "") or ""
            tags = [t for t in tag_str.split("|") if t and t != "none"]
            for cat in tags:
                bucket = category_arch_or_impl(cat)
                buckets[(cat, bucket)].append(
                    {
                        "iters": r["code_iters"],
                        "tokens": r.get("approx_tokens", 0) or 0,
                        "dur": r["duration_seconds"],
                        "source": src_name,
                    }
                )

    rows = []
    for cat in all_categories() + ["?", "none"]:
        for bucket in ("architectural", "implementation", "n/a"):
            cell = buckets.get((cat, bucket), [])
            n = len(cell)
            if n == 0 and cat in (set(all_categories()) - {"C"}):
                rows.append(
                    {
                        "root_cause": cat,
                        "bucket": bucket,
                        "count": 0,
                        "mean_iters": "",
                        "median_iters": "",
                        "mean_tokens": "",
                        "mean_duration_s": "",
                        "by_source": "",
                    }
                )
                continue
            if n == 0:
                continue
            iters_vals = [c["iters"] for c in cell]
            toks_vals = [c["tokens"] for c in cell]
            durs_vals = [c["dur"] for c in cell]
            from collections import Counter

            src_counts = Counter(c["source"] for c in cell)
            rows.append(
                {
                    "root_cause": cat,
                    "bucket": bucket,
                    "count": n,
                    "mean_iters": round(mean(iters_vals), 2),
                    "median_iters": median(iters_vals),
                    "mean_tokens": round(mean(toks_vals), 0),
                    "mean_duration_s": round(mean(durs_vals), 1),
                    "by_source": ";".join(f"{k}={v}" for k, v in sorted(src_counts.items())),
                }
            )
    return rows


# ---------------------------------------------------------------------------


def write_user_task_index(rows: list[dict], out_path: Path) -> int:
    """Re-emit index.csv filtered to the 640 user-task runs only."""
    user_rows = [r for r in rows if is_user_task_row(r)]
    user_rows.sort(
        key=lambda r: (
            r["config"],
            int(r["user_task"].split("_")[-1]),
            r.get("injection_task") or "",
        )
    )
    fields = list(user_rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in user_rows:
            w.writerow(r)
    return len(user_rows)


def main():
    rows = load_index()
    user_rows = [r for r in rows if is_user_task_row(r)]
    print(f"Loaded {len(rows)} index rows ({len(user_rows)} user-task rows)")

    out_dir = HERE
    n_idx = write_user_task_index(rows, out_dir / "index.csv")
    print(f"  index.csv:                    {n_idx} rows (user-task runs only)")
    failures_baseline = build_failures_baseline(rows)
    failures_secpol = build_failures_secpol(rows)
    failures_attack = build_failures_under_attack(rows)
    sample_passed = build_passed_expensive_sample(rows)

    base_fields = PER_TASK_FIELDS + ["arch_or_impl", "user_prompt", "code_excerpt"]
    write_csv(out_dir / "failures_baseline.csv", failures_baseline, base_fields)
    write_csv(
        out_dir / "failures_secpol.csv",
        failures_secpol,
        base_fields + ["secpol_subcause"],
    )
    write_csv(
        out_dir / "failures_under_attack.csv",
        failures_attack,
        base_fields + ["injection_effect", "injection_effect_confidence"],
    )

    sample_fields = [
        "task_id",
        "config",
        "injection_task",
        "utility",
        "security",
        "code_iters",
        "total_tool_calls",
        "duration_seconds",
        "approx_tokens",
        "applicable_root_causes",
        "primary_cause",
        "primary_cause_arch_impl",
        "classification_confidence",
        "stratum",
        "narrative",
    ]
    write_csv(out_dir / "sample_passed_expensive.csv", sample_passed, sample_fields)

    agg_rows = build_aggregate(failures_baseline, failures_secpol, failures_attack, sample_passed)
    agg_fields = [
        "root_cause",
        "bucket",
        "count",
        "mean_iters",
        "median_iters",
        "mean_tokens",
        "mean_duration_s",
        "by_source",
    ]
    write_csv(out_dir / "aggregate.csv", agg_rows, agg_fields)

    print(f"  failures_baseline.csv:        {len(failures_baseline)} rows")
    print(f"  failures_secpol.csv:          {len(failures_secpol)} rows (regressions only)")
    print(f"  failures_under_attack.csv:    {len(failures_attack)} rows")
    print(f"  sample_passed_expensive.csv:  {len(sample_passed)} rows")
    print(f"  aggregate.csv:                {len(agg_rows)} rows")


if __name__ == "__main__":
    main()
