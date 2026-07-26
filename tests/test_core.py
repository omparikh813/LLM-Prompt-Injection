import pytest

from injector.core import AuthorizationError, build_converter_variants, load_payloads, require_authorization


def test_require_authorization_raises_when_unset():
    with pytest.raises(AuthorizationError):
        require_authorization({"authorization": {"i_am_authorized_to_test_this_target": False}})


def test_require_authorization_raises_when_missing():
    with pytest.raises(AuthorizationError):
        require_authorization({})


def test_require_authorization_passes_when_true():
    require_authorization({"authorization": {"i_am_authorized_to_test_this_target": True}})


def test_build_converter_variants_none_is_empty_chain():
    variants = build_converter_variants(["none"])
    assert variants["none"] == []


def test_build_converter_variants_known_converters():
    variants = build_converter_variants(["none", "base64", "rot13"])
    assert set(variants) == {"none", "base64", "rot13"}
    assert len(variants["base64"]) == 1
    assert len(variants["rot13"]) == 1


def test_build_converter_variants_rejects_unknown_name():
    with pytest.raises(ValueError):
        build_converter_variants(["not-a-real-converter"])


def test_load_payloads_filters_by_category(tmp_path):
    payload_file = tmp_path / "payloads.yaml"
    payload_file.write_text(
        "direct_injection:\n"
        "  - id: di-001\n"
        "    prompt: hello\n"
        "    success_description: it worked\n"
        "jailbreak:\n"
        "  - id: jb-001\n"
        "    prompt: hi\n"
        "    success_description: it worked\n"
    )

    result = load_payloads(payload_file, ["direct_injection"])

    assert list(result) == ["direct_injection"]
    assert result["direct_injection"][0]["id"] == "di-001"


def test_load_payloads_ignores_missing_category(tmp_path):
    payload_file = tmp_path / "payloads.yaml"
    payload_file.write_text("direct_injection:\n  - id: di-001\n    prompt: hello\n    success_description: x\n")

    result = load_payloads(payload_file, ["direct_injection", "does_not_exist"])

    assert list(result) == ["direct_injection"]
