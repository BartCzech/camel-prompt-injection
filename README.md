# `CaMeL`: [Defeating Prompt Injections by Design](https://arxiv.org/abs/2503.18813)

Edoardo Debenedetti<sup>1,3</sup>, Ilia Shumailov<sup>2</sup>, Tianqi Fan<sup>1</sup>, Jamie Hayes<sup>2</sup>, Nicholas Carlini<sup>2</sup>, Daniel Fabian<sup>1</sup>, Christoph Kern<sup>1</sup>, Chongyang Shi<sup>2</sup>, Florian Tramèr<sup>3</sup>

<sup>1</sup>Google, <sup>2</sup>Google DeepMind, and <sup>3</sup>ETH Zurich

> [!NOTE]
> **This is a research fork.** The scope-guarded branching mechanism, DDC-3 static analyser, and evaluation harness in this repository were added by Bartłomiej Czech as part of a master's thesis.  The upstream codebase is [google-research/camel-prompt-injection](https://github.com/google-research/camel-prompt-injection) (Copyright 2025 Google LLC, Apache License 2.0).  All original upstream files are reproduced unmodified or with clearly marked changes; see `NOTICE` and individual file headers.

> [!WARNING]
> This is a research artifact released to reproduce the results in our paper. The interpreter implementation likely contains bugs (e.g., it might throw uncaught exceptions and crash) and the implementation might not be fully secure.
>
> This is **not** a Google product, and we are not planning to provide support for and/or maintain this codebase.

## Pre-requisites

1. Install `uv` via the [official instructions](https://docs.astral.sh/uv/getting-started/installation/).
2. Rename `.env.example` to `.env` and populate it with your API keys.
3. `uv` will install all dependencies as soon as you run `uv run ...`.

## Running the defense against AgentDojo

```bash
uv run main.py MODEL_NAME [--use-original] [--ad_defense] [--reasoning-effort] [--thinking_budget_tokens] [--run-attack] [--replay-with-policies] [--eval_mode]
```

The entry-point scripts load a `.env` file automatically via `python-dotenv`, so your API
keys are picked up without any extra flag. More details on the various CLI arguments can be
found by running `uv run main.py --help`.

## Thesis extensions: scope-guarded branching

This fork extends CaMeL with **scope-guarded branching**: a mechanism that lets the
privileged LLM (P-LLM) classify untrusted data into one of a fixed set of pre-declared
branches and bind each branch to an explicit allow-list of tools. The classification itself
is delegated to the quarantined LLM (Q-LLM), but the set of actions the agent can take is
fixed *before* any untrusted content is read, so a prompt injection can at worst steer the
agent into a wrong-but-declared branch — it can never expand the action set or smuggle
attacker-controlled arguments into a tool call. For patterns that cannot be expressed this
way (data-dependent control flow that derives a tool argument from tainted data, i.e. DDC-3
and DDC-4), the P-LLM calls `decline()`; an optional DDC-3 static analyser enforces this
automatically by rewriting unsafe tainted-argument calls.

This fork adds the mechanism above, a DDC-3 static analyser, and the evaluation harness
described in the accompanying thesis. The full reproduction recipe and data-artefact map are
in **Appendix A** of the thesis; the commands below are the entry points.

```bash
# Baseline reproduction (Chapter 5): camel_baseline / under_attack / secpol
uv run python main.py --model anthropic:claude-sonnet-4-20250514 --suites workspace
uv run python main.py --model anthropic:claude-sonnet-4-20250514 --suites workspace --run-attack
uv run python main.py --model anthropic:claude-sonnet-4-20250514 --suites workspace --replay-with-policies

# Full 40-task suite with scope-guarded branching enabled (Finding 10)
uv run python main.py --model anthropic:claude-sonnet-4-20250514 --suites workspace --enable-scope-guard

# Seven-scenario scope-guard evaluation (Chapter 7)
uv run python run_scenarios.py                 # full evaluation
uv run python run_scenarios.py --ddc1-only     # DDC-1 / DDC-2 only
uv run python run_scenarios.py --ddc3-only --ddc3-analyser   # DDC-3 with the static analyser
uv run python run_scenarios.py --q-llm anthropic:claude-haiku-4-5-20251001   # cross-model probe

# Automated adversarial payload corpus (Finding 8)
uv run python generate_payloads.py
uv run python run_scenarios.py --ddc1-only --payload-file payloads/

# Smoke tests (no API key required) and analyser unit tests
uv run python tests/fast_test.py
uv run python tests/test_ddc3_analyser.py
```

The failure-decomposition tagging scripts live under `thesis_analysis/`; the scope-guard
implementation is in `src/camel/scope_guard/`.

## FAQ

> How do I try a new/different model?

You can add it to the [`models.py`](src/camel/models.py) file, in the `_supported_model_names` variable. The keys are the model names with the given provider (check the provider's API) and the values is what the model says when asked "what model are you?". Keep in mind that OpenAI reasoning models are stored in the `_oai_thinking_models` variable instead.

> If I have questions on the codebase how can I reach out?

Please open an issue in this repository. Please note that we are not planning to fix bugs as this codebase is just meant as a research artifact.

## Running tests and linters

```bash
uv run ruff check --fix
uv run format
uv run pyright
uv run pytest
```

This is not an officially supported Google product. This project is not
eligible for the [Google Open Source Software Vulnerability Rewards
Program](https://bughunters.google.com/open-source-security).
