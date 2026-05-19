"""Tests for OllamaProvider error translation.

Uses dummy exception classes that duck-type ollama.ResponseError so the tests
work whether or not the real ollama package is installed.
"""

from __future__ import annotations

import pytest

from hrag.providers.llm import LLMProviderError, _translate_ollama_error


# ---------------------------------------------------------------------------
# Dummy exception classes (duck-type ollama.ResponseError)
# ---------------------------------------------------------------------------


class _FakeResponseError(Exception):
    """Mimics ollama.ResponseError with .status_code and .error attributes."""

    def __init__(self, status_code: int, error: str = "") -> None:
        self.status_code = status_code
        self.error = error
        super().__init__(f"(status code: {status_code})")


# ---------------------------------------------------------------------------
# _translate_ollama_error — unit tests
# ---------------------------------------------------------------------------


def test_503_raises_unavailable():
    exc = _FakeResponseError(503)
    result = _translate_ollama_error(exc, "llama3")
    assert isinstance(result, LLMProviderError)
    assert "unavailable" in str(result).lower()
    assert "503" in str(result)


def test_404_raises_model_not_pulled():
    exc = _FakeResponseError(404, error="model not found")
    result = _translate_ollama_error(exc, "mistral")
    assert isinstance(result, LLMProviderError)
    assert "mistral" in str(result)
    assert "ollama pull" in str(result)


def test_404_status_alone_raises_model_not_pulled():
    # Even without the "model not found" text, a bare 404 should trigger the
    # model-not-pulled message.
    exc = _FakeResponseError(404)
    result = _translate_ollama_error(exc, "my-model")
    assert isinstance(result, LLMProviderError)
    assert "my-model" in str(result)
    assert "ollama pull" in str(result)


def test_no_such_model_text_triggers_pull_hint():
    exc = _FakeResponseError(400, error="no such model: gemma2")
    result = _translate_ollama_error(exc, "gemma2")
    assert isinstance(result, LLMProviderError)
    assert "gemma2" in str(result)
    assert "ollama pull" in str(result)


def test_connection_refused_raises_cannot_reach():
    exc = ConnectionError("Connection refused")
    result = _translate_ollama_error(exc, "llama3")
    assert isinstance(result, LLMProviderError)
    assert "ollama serve" in str(result).lower() or "cannot reach" in str(result).lower()


def test_failed_to_establish_connection_raises_cannot_reach():
    exc = OSError("Failed to establish a new connection: [Errno 111] Connection refused")
    result = _translate_ollama_error(exc, "llama3")
    assert isinstance(result, LLMProviderError)
    assert "ollama serve" in str(result).lower() or "cannot reach" in str(result).lower()


def test_unrelated_exception_falls_back_to_generic():
    exc = RuntimeError("some unexpected problem")
    result = _translate_ollama_error(exc, "llama3")
    assert isinstance(result, LLMProviderError)
    assert "ollama call failed" in str(result).lower()
    assert "some unexpected problem" in str(result)


# ---------------------------------------------------------------------------
# OllamaProvider.generate — integration-style (uses stub client)
# ---------------------------------------------------------------------------


def _make_provider(client_factory=None):
    """Build an OllamaProvider with a patched _client."""
    from hrag.config import LLMConfig
    from hrag.providers.llm import OllamaProvider

    cfg = LLMConfig(provider="ollama", model="test-model")
    provider = OllamaProvider(cfg)
    if client_factory is not None:
        provider._client = client_factory()
    return provider


class _RaisingClient:
    """Fake ollama.Client that raises a given exception on .chat()."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def chat(self, **kwargs):  # noqa: ANN002
        raise self._exc


def test_generate_wraps_503():
    from hrag.types import GenerationRequest, Message

    provider = _make_provider()
    provider._client = _RaisingClient(_FakeResponseError(503))

    with pytest.raises(LLMProviderError, match="unavailable"):
        provider.generate(GenerationRequest(messages=[Message(role="user", content="hi")]))


def test_generate_wraps_404():
    from hrag.types import GenerationRequest, Message

    provider = _make_provider()
    provider._client = _RaisingClient(_FakeResponseError(404, "model not found"))

    with pytest.raises(LLMProviderError, match="ollama pull"):
        provider.generate(GenerationRequest(messages=[Message(role="user", content="hi")]))


def test_generate_stream_wraps_connection_error():
    from hrag.types import GenerationRequest, Message

    provider = _make_provider()
    provider._client = _RaisingClient(ConnectionError("Connection refused"))

    with pytest.raises(LLMProviderError, match="[Cc]annot reach|ollama serve"):
        # Consume the generator to trigger the exception.
        list(provider.generate_stream(GenerationRequest(messages=[Message(role="user", content="hi")])))


def test_exception_is_chained():
    """The original exception must be chained so tracebacks remain useful."""
    from hrag.types import GenerationRequest, Message

    original = _FakeResponseError(503)
    provider = _make_provider()
    provider._client = _RaisingClient(original)

    with pytest.raises(LLMProviderError) as exc_info:
        provider.generate(GenerationRequest(messages=[Message(role="user", content="hi")]))

    assert exc_info.value.__cause__ is original


# ---------------------------------------------------------------------------
# verify_ready — OllamaProvider
# ---------------------------------------------------------------------------


class _ModelEntry:
    """Minimal stub for ollama model list entries (duck-types the real SDK object)."""

    def __init__(self, model: str) -> None:
        self.model = model


class _ListResponse:
    """Stub for the object returned by ollama.Client().list()."""

    def __init__(self, model_names: list[str]) -> None:
        self.models = [_ModelEntry(n) for n in model_names]


class _ListingClient:
    """Fake ollama.Client whose .list() returns a controllable model list."""

    def __init__(self, model_names: list[str]) -> None:
        self._response = _ListResponse(model_names)

    def list(self):
        return self._response


class _RaisingListClient:
    """Fake ollama.Client whose .list() raises a given exception."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def list(self):
        raise self._exc


def test_ollama_verify_ready_succeeds_when_model_present():
    """verify_ready returns None when the configured model is in the list."""
    provider = _make_provider()
    provider._client = _ListingClient(["test-model", "other-model"])
    # Should not raise.
    result = provider.verify_ready()
    assert result is None


def test_ollama_verify_ready_raises_when_model_missing():
    """verify_ready raises LLMProviderError with ollama pull hint when model is absent."""
    provider = _make_provider()
    provider._client = _ListingClient(["llama3", "mistral"])

    with pytest.raises(LLMProviderError) as exc_info:
        provider.verify_ready()

    msg = str(exc_info.value)
    assert "ollama pull test-model" in msg


def test_ollama_verify_ready_raises_on_connection_error():
    """verify_ready raises LLMProviderError with 'Cannot reach Ollama' on ConnectionError."""
    provider = _make_provider()
    provider._client = _RaisingListClient(ConnectionError("Connection refused"))

    with pytest.raises(LLMProviderError) as exc_info:
        provider.verify_ready()

    assert "Cannot reach Ollama" in str(exc_info.value)


# ---------------------------------------------------------------------------
# verify_ready — default (base class) implementation
# ---------------------------------------------------------------------------


class _StubProvider:
    """Minimal concrete LLMProvider that exposes the default verify_ready."""

    def __init__(self, complete_fn):
        from hrag.config import LLMConfig
        self.config = LLMConfig(provider="stub", model="stub-model")
        self._complete_fn = complete_fn

    def complete(self, prompt, max_tokens=None, temperature=None, system=None):
        return self._complete_fn(prompt, max_tokens=max_tokens, temperature=temperature)

    # Pull in the real default verify_ready from the base class.
    verify_ready = __import__("hrag.providers.llm", fromlist=["LLMProvider"]).LLMProvider.verify_ready


def test_default_verify_ready_calls_complete():
    """The default verify_ready implementation must call complete with max_tokens=1."""
    from unittest.mock import MagicMock

    mock_complete = MagicMock(return_value="ok")
    provider = _StubProvider(mock_complete)
    provider.verify_ready()

    mock_complete.assert_called_once()
    _, kwargs = mock_complete.call_args
    assert kwargs.get("max_tokens") == 1
