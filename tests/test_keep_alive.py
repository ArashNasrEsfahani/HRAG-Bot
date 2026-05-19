"""Tests for keep_alive plumbing through OllamaProvider._build_chat_kwargs.

Uses a MagicMock client so no real Ollama server is needed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from hrag.config import LLMConfig
from hrag.providers.llm import OllamaProvider
from hrag.types import GenerationRequest, Message


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provider(keep_alive_value):
    """Build an OllamaProvider with a mock client and the given keep_alive."""
    config = LLMConfig(keep_alive=keep_alive_value)
    with patch("ollama.Client"):
        provider = OllamaProvider(config)
    provider._client = MagicMock()
    # Set up a fake response object the provider can index into
    fake_resp = {"message": {"content": "hello"}}
    provider._client.chat.return_value = fake_resp
    return provider


def _build_kwargs(provider):
    """Call _build_chat_kwargs with a minimal messages list and empty options."""
    msgs = [{"role": "user", "content": "hi"}]
    return provider._build_chat_kwargs(msgs, {})


# ---------------------------------------------------------------------------
# Tests: _build_chat_kwargs
# ---------------------------------------------------------------------------


def test_default_keep_alive_is_30m():
    """Default LLMConfig has keep_alive='30m'; must appear in chat kwargs."""
    provider = _make_provider("30m")
    kwargs = _build_kwargs(provider)
    assert kwargs.get("keep_alive") == "30m"


def test_explicit_1h_propagates():
    """Explicit keep_alive='1h' propagates into the kwargs."""
    provider = _make_provider("1h")
    kwargs = _build_kwargs(provider)
    assert kwargs.get("keep_alive") == "1h"


def test_none_keep_alive_absent_from_kwargs():
    """When keep_alive=None the key must NOT appear in chat kwargs."""
    provider = _make_provider(None)
    kwargs = _build_kwargs(provider)
    assert "keep_alive" not in kwargs


def test_never_unload_minus_1s():
    """keep_alive='-1s' (never unload) propagates correctly."""
    provider = _make_provider("-1s")
    kwargs = _build_kwargs(provider)
    assert kwargs.get("keep_alive") == "-1s"


# ---------------------------------------------------------------------------
# Test: streaming path also gets keep_alive
# ---------------------------------------------------------------------------


def test_generate_stream_sets_keep_alive():
    """generate_stream() passes keep_alive to client.chat."""
    provider = _make_provider("30m")

    # Set up mock to return an iterable of chunk dicts
    fake_chunks = [{"message": {"content": "tok"}}]
    provider._client.chat.return_value = iter(fake_chunks)

    request = GenerationRequest(messages=[Message(role="user", content="hi")])
    # Consume the generator fully
    list(provider.generate_stream(request))

    call_kwargs = provider._client.chat.call_args[1]
    assert call_kwargs.get("keep_alive") == "30m"
    # streaming flag must also be set
    assert call_kwargs.get("stream") is True
