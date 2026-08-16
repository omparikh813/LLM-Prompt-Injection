from injector.findings import Finding, RunSummary, severity_for
from injector.report import build_pdf, redact


def _make_finding(category="direct_injection", successes=0, trials=5, error_count=0):
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
        confidence="auto-scored" if trials > 0 else "no-data",
        error_count=error_count,
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


def test_finding_attempted_includes_errors_but_success_rate_excludes_them():
    # 2 scored trials (1 success) + 3 transport errors that never got a response.
    finding = _make_finding(successes=1, trials=2, error_count=3)
    assert finding.attempted == 5
    assert finding.success_rate == 0.5  # denominator is 2 (scored), not 5 (attempted)


def test_finding_severity_is_unknown_when_nothing_was_scored():
    # All attempts failed with transport errors — this must never read as
    # "Info" (which means "tested, no vulnerability found").
    finding = _make_finding(successes=0, trials=0, error_count=5)
    assert finding.trials == 0
    assert finding.attempted == 5
    assert finding.severity == "Unknown"


def test_finding_severity_is_unknown_even_for_benign_baseline_with_no_data():
    finding = _make_finding(category="benign_baseline", successes=0, trials=0, error_count=5)
    assert finding.severity == "Unknown"


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
    run_summary = RunSummary(
        findings=findings, combos_total=2, combos_completed=2, stopped_early=False, stop_reason=None
    )
    config = {"report": {"redact_critical_raw_output": True}}
    output_path = tmp_path / "report.pdf"

    result_path = build_pdf(run_summary, config, "test-model", output_path)

    assert result_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_build_pdf_renders_unknown_severity_finding(tmp_path):
    findings = [
        _make_finding(category="direct_injection", successes=0, trials=0, error_count=5),
        _make_finding(category="jailbreak", successes=2, trials=5),
    ]
    run_summary = RunSummary(
        findings=findings, combos_total=2, combos_completed=2, stopped_early=False, stop_reason=None
    )
    config = {"report": {"redact_critical_raw_output": True}}
    output_path = tmp_path / "report.pdf"

    build_pdf(run_summary, config, "test-model", output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_build_pdf_shows_partial_run_banner(tmp_path):
    findings = [_make_finding(category="direct_injection", successes=1, trials=1)]
    run_summary = RunSummary(
        findings=findings,
        combos_total=10,
        combos_completed=1,
        stopped_early=True,
        stop_reason="Target API rate limit exceeded at combination 2/10",
    )
    config = {"report": {"redact_critical_raw_output": True}}
    output_path = tmp_path / "report.pdf"

    build_pdf(run_summary, config, "test-model", output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
