"""PDF report generation from findings (Jinja2 HTML template -> WeasyPrint)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML

from .findings import RunSummary

TEMPLATE_DIR = Path(__file__).parent / "templates"

# Raw evidence for Critical/High findings is truncated in the PDF body by
# default — the report is a restricted-distribution artifact in its own
# right (see PRD.md section 16), and full untruncated evidence stays in the
# JSON findings log rather than the document handed around.
REDACT_SEVERITIES = {"Critical", "High"}
MAX_RESPONSE_CHARS = 500

SEVERITY_ORDER = ["Critical", "High", "Medium", "Low", "Info"]


def redact(text: str, severity: str, enabled: bool) -> str:
    if not enabled or severity not in REDACT_SEVERITIES or len(text) <= MAX_RESPONSE_CHARS:
        return text
    return text[:MAX_RESPONSE_CHARS] + "\n\n[REDACTED — full evidence retained in the local findings log only]"


def build_pdf(run_summary: RunSummary, config: dict[str, Any], model_id: str, output_path: str | Path) -> Path:
    findings = run_summary.findings
    report_cfg = config.get("report", {})
    redact_enabled = report_cfg.get("redact_critical_raw_output", True)

    severity_counts = Counter(f.severity for f in findings)
    rows = [
        {
            "payload_id": f.payload_id,
            "category": f.category,
            "converter": f.converter,
            "severity": f.severity,
            "success_rate_display": f"{f.success_rate:.0%} ({f.successes}/{f.trials})",
            "prompt_sent": f.prompt_sent,
            "sample_response": redact(f.sample_response, f.severity, redact_enabled),
            "owasp": f.framework["owasp"],
            "atlas": f.framework["atlas"],
            "confidence": f.confidence,
            "error_count": f.error_count,
        }
        for f in sorted(findings, key=lambda x: (SEVERITY_ORDER.index(x.severity), -x.success_rate))
    ]

    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)), autoescape=True)
    template = env.get_template("report.html.jinja2")
    html = template.render(
        model_id=model_id,
        tested_at=findings[0].tested_at if findings else "n/a",
        severity_order=SEVERITY_ORDER,
        severity_counts=severity_counts,
        rows=rows,
        total_findings=len(findings),
        combos_total=run_summary.combos_total,
        combos_completed=run_summary.combos_completed,
        stopped_early=run_summary.stopped_early,
        stop_reason=run_summary.stop_reason,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    HTML(string=html).write_pdf(str(output_path))
    return output_path
