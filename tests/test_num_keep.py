"""Tests for num_keep plumbing through OllamaProvider._build_options.

num_keep is an Ollama *options* field (nested under "options" in the HTTP
request body), NOT a top-level chat kwarg like keep_alive or think.

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


def _make_provider(**kwargs):
    """Build an OllamaProvider with a mock client and the given LLMConfig kwargs."""
    config = LLMConfig(**kwargs)
    with patch("ollama.Client"):
        provider = OllamaProvider(config)
    provider._client = MagicMock()
    fake_resp = {"message": {"content": "hello"}}
    provider._client.chat.return_value = fake_resp
    return provider


def _minimal_request():
    return GenerationRequest(messages=[Message(role="user", content="hi")])


# ---------------------------------------------------------------------------
# Tests: _build_options (the key placement)
# ---------------------------------------------------------------------------


def test_default_num_keep_absent_from_options():
    """Default LLMConfig (num_keep=None) must NOT add 'num_keep' to options."""
    provider = _make_provider()
    opts = provider._build_options(_minimal_request())
    assert "num_keep" not in opts


def test_explicit_num_keep_256_in_options():
    """LLMConfig(num_keep=256) must put num_keep=256 inside options."""
    provider = _make_provider(num_keep=256)
    opts = provider._build_options(_minimal_request())
    assert opts["num_keep"] == 256


def test_num_keep_zero_absent_from_options():
    """num_keep=0 means 'use server default'; must NOT appear in options."""
    provider = _make_provider(num_keep=0)
    opts = provider._build_options(_minimal_request())
    assert "num_keep" not in opts


def test_generate_stream_threads_num_keep():
    """generate_stream() must pass num_keep inside options to client.chat."""
    provider = _make_provider(num_keep=512)

    fake_chunks = [{"message": {"content": "tok"}}]
    provider._client.chat.return_value = iter(fake_chunks)

    list(provider.generate_stream(_minimal_request()))

    call_kwargs = provider._client.chat.call_args[1]
    assert call_kwargs["options"]["num_keep"] == 512
    assert call_kwargs.get("stream") is True


def test_num_keep_in_options_keep_alive_at_top_level():
    """num_keep=128 AND keep_alive='1h': former in options, latter at top level."""
    provider = _make_provider(num_keep=128, keep_alive="1h")

    # non-streaming path
    fake_resp = {"message": {"content": "hello"}}
    provider._client.chat.return_value = fake_resp

    provider.generate(_minimal_request())

    call_kwargs = provider._client.chat.call_args[1]
    # num_keep must be nested inside options
    assert call_kwargs["options"]["num_keep"] == 128
    # keep_alive must be a top-level kwarg, NOT inside options
    assert call_kwargs.get("keep_alive") == "1h"
    assert "keep_alive" not in call_kwargs["options"]
