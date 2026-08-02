"""Integration-style tests for run_attacks' progress/early-stop control flow.

PromptSendingAttack.execute_async is monkeypatched so these run fully
offline against dummy credentials — they exercise our own orchestration
logic in findings.py, not PyRIT's network layer (which is validated
separately by the installed-package import checks elsewhere in this suite).
"""

from types import SimpleNamespace

import httpx
import pytest
from openai import RateLimitError

from injector.findings import run_attacks


class _FakeScore:
    def __init__(self, value: bool):
        self._value = value
        self.score_rationale = f"fake rationale: {value}"

    def get_value(self) -> bool:
        return self._value


def _fake_result(success: bool):
    return SimpleNamespace(last_score=_FakeScore(success))


@pytest.fixture
def dummy_env(monkeypatch):
    monkeypatch.setenv("TARGET_ENDPOINT", "https://example-target.test/v1")
    monkeypatch.setenv("TARGET_API_KEY", "dummy-target-key")
    monkeypatch.setenv("TARGET_MODEL", "dummy-target-model")
    monkeypatch.setenv("JUDGE_ENDPOINT", "https://example-judge.test/v1")
    monkeypatch.setenv("JUDGE_API_KEY", "dummy-judge-key")
    monkeypatch.setenv("JUDGE_MODEL", "dummy-judge-model")


def _config():
    return {
        "target": {
            "name": "dummy-target",
            "endpoint_env": "TARGET_ENDPOINT",
            "api_key_env": "TARGET_API_KEY",
            "model_env": "TARGET_MODEL",
            "max_requests_per_minute": 1000,
        },
        "judge": {
            "endpoint_env": "JUDGE_ENDPOINT",
            "api_key_env": "JUDGE_API_KEY",
            "model_env": "JUDGE_MODEL",
        },
        "run": {
            "trials_per_payload": 2,
            "max_attempts_total": 500,
            "converters": ["none"],
        },
    }


def _payloads():
    return {
        "direct_injection": [
            {"id": "di-001", "prompt": "p1", "success_description": "s1"},
            {"id": "di-002", "prompt": "p2", "success_description": "s2"},
        ],
    }


@pytest.mark.asyncio
async def test_run_attacks_completes_normally(dummy_env, monkeypatch):
    async def fake_execute_async(self, *, objective, **kwargs):
        return _fake_result(success=True)

    monkeypatch.setattr("pyrit.executor.attack.PromptSendingAttack.execute_async", fake_execute_async)

    run_summary = await run_attacks(_config(), _payloads())

    assert run_summary.stopped_early is False
    assert run_summary.combos_total == 2
    assert run_summary.combos_completed == 2
    assert all(f.successes == f.trials for f in run_summary.findings)


@pytest.mark.asyncio
async def test_run_attacks_stops_early_on_rate_limit_and_keeps_partial_results(dummy_env, monkeypatch):
    call_count = {"n": 0}

    async def fake_execute_async(self, *, objective, **kwargs):
        call_count["n"] += 1
        if call_count["n"] >= 3:
            response = httpx.Response(status_code=429, request=httpx.Request("POST", "https://example.com"))
            raise Exception("Error sending prompt with conversation ID: fake") from RateLimitError(
                "rate limited", response=response, body=None
            )
        return _fake_result(success=True)

    monkeypatch.setattr("pyrit.executor.attack.PromptSendingAttack.execute_async", fake_execute_async)

    run_summary = await run_attacks(_config(), _payloads())

    assert run_summary.stopped_early is True
    assert "rate limit" in run_summary.stop_reason.lower()
    assert run_summary.combos_total == 2
    assert run_summary.combos_completed < run_summary.combos_total
    # The 2 successful trials before the rate limit hit are not discarded.
    assert run_summary.findings[0].trials == 2
    assert run_summary.findings[0].successes == 2
