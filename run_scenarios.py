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

"""Scope-guard DDC evaluation runner.

Runs S1–S7 scenarios against ScopedPrivilegedLLM and records utility, security,
and decline metrics per scenario × adversarial payload combination.

Usage
-----
    uv run python run_scenarios.py                    # all scenarios, Sonnet 4
    uv run python run_scenarios.py --scenarios S1 S2  # subset
    uv run python run_scenarios.py --model anthropic:claude-3-5-haiku-20241022
    uv run python run_scenarios.py --output results/scope_eval.csv

Output columns
--------------
    scenario      S1 … S7
    category      DDC-1 / DDC-2 / DDC-3
    payload       name of the adversarial payload ("benign" for the baseline)
    utility       1 = agent completed/declined correctly, 0 = wrong action
    security      1 = attacker did NOT succeed, 0 = attacker succeeded
    decline       1 = agent called decline(), 0 = no decline detected
    state_changed 1 = post-environment differs from pre-environment
    error         non-empty string if an unexpected exception occurred
"""

from __future__ import annotations

import csv
import dataclasses
import json
import os
import re
import sys
import traceback
from pathlib import Path

import anthropic
import cyclopts
from agentdojo import agent_pipeline, functions_runtime
from agentdojo.task_suite import get_suite
from dotenv import load_dotenv

import camel.custom_yaml  # noqa: F401 — registers YAML constructors

load_dotenv()

from camel.pipeline_elements.security_policies import ADNoSecurityPolicyEngine
from camel.scope_guard.pipeline import ScopedPrivilegedLLM
from scenarios.workspace_ddc_tasks import ALL_SCENARIOS, DDC1_SCENARIOS, DDC3_SCENARIOS, Scenario

@dataclasses.dataclass
class RunResult:
    scenario: str
    category: str
    payload: str
    utility: int
    security: int
    decline: int
    state_changed: int
    error: str = ""


def make_scoped_pipeline(
    model: str,
    q_llm: str | None = None,
    ddc3_analyser: bool = False,
) -> ScopedPrivilegedLLM:
    """Build a ScopedPrivilegedLLM using Anthropic as backend."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY not set")

    client = anthropic.Anthropic(api_key=api_key)
    model_name = model.split(":")[-1]  # strip "anthropic:" prefix if present
    llm = agent_pipeline.AnthropicLLM(client, model_name, max_tokens=8192)
    llm.name = model_name

    return ScopedPrivilegedLLM(
        llm=llm,
        security_policy_engine=ADNoSecurityPolicyEngine,
        quarantined_llm_model=f"anthropic:{q_llm or model_name}",
        max_attempts=3,
        ddc3_analyser=ddc3_analyser,
    )


def _dump_trace(
    trace_dir: Path,
    scenario_id: str,
    payload_name: str,
    messages: list,
) -> None:
    """Persist raw messages and extracted Python code blocks to trace_dir.

    Writes two files per run:
      <scenario>_<payload>.json   — serialised message list (str repr of each msg)
      <scenario>_<payload>.py     — only the Python code blocks the P-LLM emitted,
                                    one block per iteration separated by a header
                                    comment. This is the file to read for Task 0.
    """
    trace_dir.mkdir(parents=True, exist_ok=True)
    safe_payload = re.sub(r"[^A-Za-z0-9_-]", "_", payload_name)
    stem = f"{scenario_id}_{safe_payload}"

    msg_path = trace_dir / f"{stem}.json"
    try:
        serialisable = [str(m) for m in messages]
        msg_path.write_text(json.dumps(serialisable, indent=2, ensure_ascii=False))
    except Exception as exc:  # noqa: BLE001
        msg_path.write_text(json.dumps({"error": str(exc)}))

    code_path = trace_dir / f"{stem}.py"
    blocks: list[str] = []
    for i, msg in enumerate(messages):
        text = str(msg)
        for block in re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL):
            blocks.append(f"# --- iteration {i} ---\n{block.rstrip()}")
    code_path.write_text("\n\n".join(blocks) if blocks else "# no python blocks found\n")


def run_once(
    scenario: Scenario,
    payload_name: str,
    payload_text: str,
    pipeline: ScopedPrivilegedLLM,
    suite,
    trace_dir: Path | None = None,
) -> RunResult:
    """Execute one scenario × payload combination and return a RunResult."""
    injections = {scenario.injection_slot: payload_text} if payload_text else {}

    try:
        env = suite.load_and_inject_default_environment(injections)
    except Exception as e:
        return RunResult(
            scenario=scenario.id,
            category=scenario.category,
            payload=payload_name,
            utility=0,
            security=0,
            decline=0,
            state_changed=0,
            error=f"env_load_error: {e}",
        )

    pre_env = env.model_copy(deep=True)
    runtime = functions_runtime.FunctionsRuntime(suite.tools)
    messages: list = []

    try:
        _, _, post_env, messages, _ = pipeline.query(scenario.prompt, runtime, env, messages)
    except Exception as e:
        tb = traceback.format_exc()
        if trace_dir:
            _dump_trace(trace_dir, scenario.id, payload_name, messages)
        return RunResult(
            scenario=scenario.id,
            category=scenario.category,
            payload=payload_name,
            utility=0,
            security=0,
            decline=0,
            state_changed=0,
            error=f"pipeline_error: {type(e).__name__}: {e!s:.200}\n{tb[:400]}",
        )

    if trace_dir:
        _dump_trace(trace_dir, scenario.id, payload_name, messages)

    try:
        state_changed = int(pre_env != post_env)
        utility = int(bool(scenario.utility_check(pre_env, post_env, messages)))
        security = int(bool(scenario.security_check(pre_env, post_env, messages)))
        decline = int(ScopedPrivilegedLLM.is_decline("", messages))
    except Exception as e:
        return RunResult(
            scenario=scenario.id,
            category=scenario.category,
            payload=payload_name,
            utility=0,
            security=1,
            decline=0,
            state_changed=0,
            error=f"grader_error: {e}",
        )

    return RunResult(
        scenario=scenario.id,
        category=scenario.category,
        payload=payload_name,
        utility=utility,
        security=security,
        decline=decline,
        state_changed=state_changed,
    )


def _load_payload_file(path: str) -> dict[str, dict[str, str]]:
    """Load a generated payload file and return {scenario_id: {payload_id: text}}.

    Accepts either a single scenario file (payloads/S1_generated.json) or a
    directory (payloads/) in which case all *_generated.json files are loaded.
    """
    p = Path(path)
    files: list[Path] = []
    if p.is_dir():
        files = sorted(p.glob("*_generated.json"))
    elif p.is_file():
        files = [p]
    else:
        print(f"Payload file/dir not found: {path}", file=sys.stderr)
        sys.exit(1)

    result: dict[str, dict[str, str]] = {}
    for f in files:
        # Infer scenario ID from filename, e.g. S1_generated.json → S1
        scenario_id = f.stem.split("_")[0]
        entries: list[dict] = json.load(f.open())
        result[scenario_id] = {e["id"]: e["text"] for e in entries}
        print(f"  Loaded {len(entries)} payloads for {scenario_id} from {f}")
    return result


def main(
    model: str = "anthropic:claude-sonnet-4-20250514",
    q_llm: str | None = None,
    output: str = "results/scope_eval.csv",
    scenarios: list[str] | None = None,
    ddc1_only: bool = False,
    ddc3_only: bool = False,
    dry_run: bool = False,
    save_traces: bool = False,
    trace_dir: str = "results/traces",
    payload_file: str | None = None,
    ddc3_analyser: bool = False,
):
    """Run the DDC scenario evaluation.

    Args:
        model: P-LLM model (provider:model_name format, e.g. anthropic:claude-sonnet-4-20250514).
        q_llm: Q-LLM model for the classifier. Defaults to the same as --model.
        output: Path to the output CSV file.
        scenarios: Whitelist of scenario IDs to run (e.g. S1 S2). Default: all.
        ddc1_only: Run only DDC-1/DDC-2 scenarios (S1–S4).
        ddc3_only: Run only DDC-3 scenarios (S5–S7).
        dry_run: Print what would run without calling any LLM.
        save_traces: Persist per-run message logs and extracted Python code blocks to
            trace_dir. Each run produces <scenario>_<payload>.json (full message list)
            and <scenario>_<payload>.py (extracted Python code blocks only).
        trace_dir: Directory for trace files when --save-traces is set.
        payload_file: Path to a generated payload JSON file or directory of such files
            (produced by generate_payloads.py). When set, replaces the hardcoded
            adversarial payloads for any matching scenario ID with the loaded set.
            Benign runs are always included regardless of this flag.
        ddc3_analyser: Enable the DDC-3 static analyser. When set, each P-LLM code
            block is inspected before interpretation: any call to a state-changing
            sink whose argument derives from query_ai_assistant() is replaced with
            decline(). Taint state is accumulated across iterations within a single
            task and reset between tasks.
    """
    if ddc1_only:
        scenario_pool = DDC1_SCENARIOS
    elif ddc3_only:
        scenario_pool = DDC3_SCENARIOS
    else:
        scenario_pool = ALL_SCENARIOS

    if scenarios:
        scenario_pool = [s for s in scenario_pool if s.id in scenarios]

    if not scenario_pool:
        print("No scenarios match the given filters.", file=sys.stderr)
        sys.exit(1)

    generated: dict[str, dict[str, str]] = {}
    if payload_file:
        print("Loading generated payloads…")
        generated = _load_payload_file(payload_file)

    work: list[tuple[Scenario, str, str]] = []
    for scenario in scenario_pool:
        # Always include the benign run
        benign_payloads: dict[str, str] = {"benign": scenario.benign_payload}
        # Use generated payloads if available for this scenario, else fall back to hardcoded
        if scenario.id in generated:
            adv_payloads: dict[str, str] = generated[scenario.id]
        else:
            adv_payloads = scenario.adversarial_payloads
        all_payloads = benign_payloads | adv_payloads
        for payload_name, payload_text in all_payloads.items():
            work.append((scenario, payload_name, payload_text))

    total = len(work)
    _trace_dir: Path | None = Path(trace_dir) if save_traces else None

    print(f"Plan: {total} runs across {len(scenario_pool)} scenarios")
    print(f"  Model: {model}  |  Q-LLM: {q_llm or model}")
    print(f"  DDC-3 analyser: {'ENABLED' if ddc3_analyser else 'disabled'}")
    print(f"  Output: {output}")
    if _trace_dir:
        print(f"  Traces: {_trace_dir}/<scenario>_<payload>.{{json,py}}")

    if dry_run:
        for i, (s, p, _) in enumerate(work):
            print(f"  [{i+1:2d}/{total}] {s.id} ({s.category})  payload={p}")
        return

    pipeline = make_scoped_pipeline(model, q_llm, ddc3_analyser=ddc3_analyser)
    suite = get_suite("v1.2", "workspace")

    results: list[RunResult] = []
    for i, (scenario, payload_name, payload_text) in enumerate(work, 1):
        print(f"[{i:2d}/{total}] {scenario.id} ({scenario.category})  payload={payload_name} ...", end="", flush=True)
        result = run_once(scenario, payload_name, payload_text, pipeline, suite, _trace_dir)
        results.append(result)
        tag = "OK" if not result.error else f"ERR({result.error[:40]})"
        print(f"  utility={result.utility} security={result.security} decline={result.decline} [{tag}]")

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(dataclasses.asdict(results[0]).keys()))
        writer.writeheader()
        writer.writerows(dataclasses.asdict(r) for r in results)
    print(f"\nResults written to {out_path} ({len(results)} rows)")

    print("\n=== SUMMARY ===")
    for scenario in scenario_pool:
        s_results = [r for r in results if r.scenario == scenario.id]
        benign = next((r for r in s_results if r.payload == "benign"), None)
        adv = [r for r in s_results if r.payload != "benign"]
        if not adv:
            continue
        classifier_flips = sum(1 for r in adv if r.utility == 0 and not r.decline and not r.error)
        sec_failures = sum(1 for r in adv if r.security == 0)
        decline_count = sum(1 for r in adv if r.decline)
        print(
            f"  {scenario.id} ({scenario.category:5s})  "
            f"benign_utility={benign.utility if benign else '?'}  "
            f"adv_classifier_flips={classifier_flips}/{len(adv)}  "
            f"adv_sec_failures={sec_failures}/{len(adv)}  "
            f"adv_declines={decline_count}/{len(adv)}"
        )


if __name__ == "__main__":
    cyclopts.run(main)
