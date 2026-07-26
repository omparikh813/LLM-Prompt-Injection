"""Config loading, the authorization gate, and PyRIT target/converter construction."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pyrit.converter import Base64Converter, Converter, LeetspeakConverter, ROT13Converter
from pyrit.prompt_target import OpenAIChatTarget

# "none" is a pseudo-converter meaning "send the payload unmodified" —
# it's kept as its own named variant so a baseline run is always comparable
# against each encoding/obfuscation variant.
CONVERTER_REGISTRY: dict[str, type[Converter] | None] = {
    "none": None,
    "base64": Base64Converter,
    "rot13": ROT13Converter,
    "leetspeak": LeetspeakConverter,
}


class AuthorizationError(RuntimeError):
    """Raised when a run is attempted without the authorization ack set."""


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def require_authorization(config: dict[str, Any]) -> None:
    if not config.get("authorization", {}).get("i_am_authorized_to_test_this_target"):
        raise AuthorizationError(
            "Refusing to run: set authorization.i_am_authorized_to_test_this_target: true "
            "in the config, and only after confirming you have the right to test this "
            "target (your own account/API key, or written client authorization)."
        )


def _env(section: dict[str, Any], key: str) -> str:
    var_name = section[key]
    try:
        return os.environ[var_name]
    except KeyError:
        raise RuntimeError(
            f"Environment variable '{var_name}' is not set (required by config key '{key}'). "
            "Copy .env.example to .env and fill it in."
        )


def build_target(config: dict[str, Any]) -> OpenAIChatTarget:
    target_cfg = config["target"]
    return OpenAIChatTarget(
        endpoint=_env(target_cfg, "endpoint_env"),
        api_key=_env(target_cfg, "api_key_env"),
        model_name=_env(target_cfg, "model_env"),
        max_requests_per_minute=target_cfg.get("max_requests_per_minute", 20),
    )


def build_judge_target(config: dict[str, Any]) -> OpenAIChatTarget:
    judge_cfg = config["judge"]
    return OpenAIChatTarget(
        endpoint=_env(judge_cfg, "endpoint_env"),
        api_key=_env(judge_cfg, "api_key_env"),
        model_name=_env(judge_cfg, "model_env"),
    )


def build_converter_variants(names: list[str]) -> dict[str, list[Converter]]:
    """Map each configured converter name to the converter chain it represents.

    Each name is its own independent variant to test (not chained together) —
    e.g. ["none", "base64"] runs every payload twice: once verbatim, once
    base64-encoded, so filter-bypass effects are visible against the baseline.
    """
    variants: dict[str, list[Converter]] = {}
    for name in names:
        if name not in CONVERTER_REGISTRY:
            raise ValueError(f"Unknown converter '{name}'. Known converters: {sorted(CONVERTER_REGISTRY)}")
        converter_cls = CONVERTER_REGISTRY[name]
        variants[name] = [converter_cls()] if converter_cls else []
    return variants


def load_payloads(path: str | Path, categories: list[str]) -> dict[str, list[dict[str, Any]]]:
    all_payloads = load_yaml(path)
    return {category: all_payloads[category] for category in categories if category in all_payloads}
