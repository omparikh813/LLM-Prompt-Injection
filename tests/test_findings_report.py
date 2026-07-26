from injector.findings import Finding, severity_for
from injector.report import build_pdf, redact


def _make_finding(category="direct_injection", successes=0, trials=5, severity_hint=None):
    return Finding(
        payload_id="di-001",
        category=category,
        converter="none",
        prompt_sent="ignore previous instructions...",
        success_description="the response complies",
        trials=trials,
        successes=successes,
        sample_response="x" * 1000,
        model_id="test-model",
        tested_at="2026-01-01 00:00:00 UTC",
    )


def test_severity_for_benign_baseline_is_always_info():
    assert severity_for("benign_baseline", 1.0) == "Info"
    assert severity_for("benign_baseline", 0.0) == "Info"


def test_severity_for_thresholds():
    assert severity_for("jailbreak", 0.0) == "Info"
    assert severity_for("jailbreak", 0.1) == "Low"
    assert severity_for("jailbreak", 0.4) == "Medium"
    assert severity_for("jailbreak", 0.7) == "High"
    assert severity_for("jailbreak", 0.9) == "Critical"


def test_finding_success_rate_and_severity_properties():
    finding = _make_finding(successes=4, trials=5)
    assert finding.success_rate == 0.8
    assert finding.severity == "Critical"


def test_redact_truncates_only_above_threshold_severities():
    long_text = "x" * 1000
    assert redact(long_text, "Critical", enabled=True) != long_text
    assert redact(long_text, "Low", enabled=True) == long_text
    assert redact(long_text, "Critical", enabled=False) == long_text


def test_redact_leaves_short_text_untouched():
    short_text = "short"
    assert redact(short_text, "Critical", enabled=True) == short_text


def test_build_pdf_generates_a_file(tmp_path):
    findings = [
        _make_finding(category="direct_injection", successes=4, trials=5),
        _make_finding(category="benign_baseline", successes=5, trials=5),
    ]
    config = {"report": {"redact_critical_raw_output": True}}
    output_path = tmp_path / "report.pdf"

    result_path = build_pdf(findings, config, "test-model", output_path)

    assert result_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0
