# LLM Prompt Injector

A [PyRIT](https://github.com/microsoft/PyRIT)-based prompt injection and
jailbreak tester for LLMs, aimed at smaller/open-source models that ship
with thinner safety tuning than frontier models. It runs a payload library
of direct injection, jailbreak, and system-prompt-leak attempts against a
target model's API (with a benign baseline for comparison), scores success
with a separate judge LLM, and produces a PDF findings report.

See `PRD.md` for the full product spec — this is the Phase 1 (MVP) build:
single-turn attacks + encoding converters + automated scoring + PDF report.
Multi-turn escalation and indirect injection are Phase 2, not yet
implemented.

## Before you run this against anything

This tool actively attacks a live model endpoint. Only point it at a
target you have the right to test — your own account/API key, or written
authorization if testing on behalf of someone else. The config has an
explicit `authorization.i_am_authorized_to_test_this_target` flag that
must be set to `true` before a run will execute. See `PRD.md` section 6.

## Setup

```bash
pip install -e ".[dev]"
cp .env.example .env        # fill in your target + judge model credentials
cp data/config.example.yaml data/config.yaml
```

Edit `data/config.yaml`:
- Set `authorization.i_am_authorized_to_test_this_target: true` (only once
  you actually have that authorization).
- Point `target` at the model you're testing (any OpenAI-compatible chat
  completions endpoint — OpenRouter, Together.ai, Groq, a local Ollama
  instance, etc.).
- Point `judge` at a separate model used only to score attack success —
  this is test infrastructure, not the system under test, so it's fine to
  use a different/stronger model here.

## Run

```bash
injector --config data/config.yaml --payloads data/payloads.yaml
```

This produces, in `reports/`:
- `<target-name>-findings.json` — full per-attempt data (auditable log).
- `<target-name>-report.pdf` — the findings report, with severity ratings
  mapped to OWASP Top 10 for LLM Applications and MITRE ATLAS.

Neither file is committed to the repo (`reports/` is gitignored) — the
report contains literal model output from successful attacks and should be
handled like any other pentest deliverable: restricted distribution, not a
public artifact.

## Adding attack payloads

Edit `data/payloads.yaml`. Each entry needs:
- `id` — stable identifier used in the report.
- `prompt` — exact text sent to the target.
- `success_description` — the "true" condition given to the judge LLM to
  decide whether this specific attack succeeded.

## Adding a target model / provider

No code changes needed — add a config block pointing `endpoint_env` /
`api_key_env` / `model_env` at whatever env vars hold that provider's
details, as long as it exposes an OpenAI-compatible chat completions API.

## A note on the PyRIT API surface

This targets PyRIT's current `pyrit.executor.attack` API generation
(`PromptSendingAttack`, `AttackScoringConfig`, etc.), confirmed against the
`microsoft/PyRIT` repository at build time. PyRIT's API has changed
significantly across versions (it previously used a `pyrit.orchestrator.*`
naming scheme) — if `pip install pyrit` gives you an older release using
that API, either upgrade to a version with the `executor.attack` module, or
adapt the imports in `src/injector/findings.py` and `core.py` accordingly.

