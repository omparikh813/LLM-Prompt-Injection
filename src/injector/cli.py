"""CLI entry point: load config + payloads, run attacks, generate the PDF report."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

from .core import AuthorizationError, load_payloads, load_yaml, require_authorization
from .findings import run_attacks, save_findings_json
from .report import build_pdf


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(prog="injector", description="PyRIT-based LLM prompt injection tester")
    parser.add_argument("--config", default="data/config.example.yaml", help="Path to run config YAML")
    parser.add_argument("--payloads", default="data/payloads.yaml", help="Path to payload library YAML")
    args = parser.parse_args()

    config = load_yaml(args.config)
    try:
        require_authorization(config)
    except AuthorizationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)

    payloads = load_payloads(args.payloads, config["run"]["attack_categories"])
    if not payloads:
        print("ERROR: no matching payload categories found — check run.attack_categories", file=sys.stderr)
        raise SystemExit(1)

    findings = asyncio.run(run_attacks(config, payloads))

    model_id = config["target"].get("name", "target")
    output_dir = Path(config.get("report", {}).get("output_dir", "reports"))
    json_path = output_dir / f"{model_id}-findings.json"
    pdf_path = output_dir / f"{model_id}-report.pdf"

    save_findings_json(findings, json_path)
    build_pdf(findings, config, model_id, pdf_path)

    print(f"Ran {len(findings)} payload/converter combinations against '{model_id}'.")
    print(f"Findings log: {json_path}")
    print(f"PDF report:   {pdf_path}")


if __name__ == "__main__":
    main()
