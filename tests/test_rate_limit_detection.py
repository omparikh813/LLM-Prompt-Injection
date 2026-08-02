import httpx
from openai import RateLimitError

from injector.findings import _is_rate_limit_error


def _make_rate_limit_error() -> RateLimitError:
    response = httpx.Response(status_code=429, request=httpx.Request("POST", "https://example.com"))
    return RateLimitError("rate limited", response=response, body=None)


def test_detects_rate_limit_error_directly():
    assert _is_rate_limit_error(_make_rate_limit_error()) is True


def test_detects_rate_limit_error_wrapped_via_cause():
    # Mirrors how PyRIT wraps the underlying SDK exception:
    # raise Exception(f"Error sending prompt with conversation ID: {cid}") from ex
    wrapped = Exception("Error sending prompt with conversation ID: abc-123")
    wrapped.__cause__ = _make_rate_limit_error()

    assert _is_rate_limit_error(wrapped) is True


def test_detects_rate_limit_error_wrapped_multiple_levels_via_context():
    inner = Exception("outer wrapper")
    inner.__context__ = _make_rate_limit_error()
    outer = Exception("outermost wrapper")
    outer.__cause__ = inner

    assert _is_rate_limit_error(outer) is True


def test_does_not_false_positive_on_unrelated_exception():
    assert _is_rate_limit_error(ValueError("something else went wrong")) is False


def test_does_not_infinite_loop_on_self_referential_cause():
    exc = Exception("weird")
    exc.__cause__ = exc  # pathological but shouldn't hang

    assert _is_rate_limit_error(exc) is False
