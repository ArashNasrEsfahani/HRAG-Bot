"""Phase 9.5 — Anthropic prompt caching tests.

Stubs the `anthropic` SDK via a fake module installed into ``sys.modules``
before importing AnthropicProvider, so these tests run anywhere — no real
network calls and no need for the SDK to be installed.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from hrag.config import LLMConfig
from hrag.types import GenerationRequest, Message


# ---------------------------------------------------------------------------
# Fake `anthropic` SDK — captures the last create() / stream() call args.
# ---------------------------------------------------------------------------


class _CapturedCall:
    """Holds the kwargs Anthropic was called with on the last invocation."""

    def __init__(self) -> None:
        self.kwargs: dict | None = None


class _FakeContent:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _FakeResponse:
    def __init__(self, text: str = "ok") -> None:
        self.content = [_FakeContent(text)]


class _FakeStreamCtx:
    def __init__(self) -> None:
        self.text_stream = iter(["ok"])

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False


class _FakeMessages:
    def __init__(self, captured: _CapturedCall) -> None:
        self._captured = captured

    def create(self, **kwargs: Any) -> _FakeResponse:
        self._captured.kwargs = kwargs
        return _FakeResponse()

    def stream(self, **kwargs: Any) -> _FakeStreamCtx:
        self._captured.kwargs = kwargs
        return _FakeStreamCtx()


class _FakeAnthropic:
    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.messages = _FakeMessages(_FakeAnthropic._captured)

    _captured = _CapturedCall()


@pytest.fixture
def fake_anthropic(monkeypatch):
    """Install a fake `anthropic` module into sys.modules.

    Returns the _CapturedCall object so the test can inspect the most-recent
    kwargs the provider passed to the SDK.
    """
    captured = _CapturedCall()

    class _FakeClient:
        def __init__(self, api_key=None, base_url=None):
            self.messages = _FakeMessages(captured)

    fake_mod = types.ModuleType("anthropic")
    fake_mod.Anthropic = _FakeClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", fake_mod)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    return captured


def _request_with(system: str, user: str) -> GenerationRequest:
    return GenerationRequest(
        messages=[
            Message(role="system", content=system),
            Message(role="user", content=user),
        ],
        temperature=0.0,
        max_tokens=32,
    )


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------


def test_anthropic_caching_default_off():
    assert LLMConfig().anthropic_prompt_caching is False


# ---------------------------------------------------------------------------
# Caching OFF — system + user pass through as plain strings (no regression)
# ---------------------------------------------------------------------------


def test_anthropic_caching_off_uses_string_system(fake_anthropic):
    from hrag.providers.llm import AnthropicProvider

    cfg = LLMConfig(provider="anthropic", model="claude-3-5-sonnet-20241022",
                    anthropic_prompt_caching=False)
    p = AnthropicProvider(cfg)
    p.generate(_request_with("x" * 4000, "y" * 4000))

    sys_arg = fake_anthropic.kwargs["system"]
    assert isinstance(sys_arg, str), f"expected string system, got {type(sys_arg).__name__}"
    user_msg = fake_anthropic.kwargs["messages"][-1]
    assert isinstance(user_msg["content"], str), (
        f"expected string user content, got {type(user_msg['content']).__name__}"
    )


# ---------------------------------------------------------------------------
# Caching ON — wraps system + last user in cache_control blocks
# ---------------------------------------------------------------------------


def test_anthropic_caching_on_wraps_system_above_threshold(fake_anthropic):
    from hrag.providers.llm import AnthropicProvider

    cfg = LLMConfig(provider="anthropic", model="claude-3-5-sonnet-20241022",
                    anthropic_prompt_caching=True)
    p = AnthropicProvider(cfg)
    big_system = "S" * 2048
    p.generate(_request_with(big_system, "hi"))

    sys_arg = fake_anthropic.kwargs["system"]
    assert isinstance(sys_arg, list)
    assert len(sys_arg) == 1
    assert sys_arg[0]["type"] == "text"
    assert sys_arg[0]["text"] == big_system
    assert sys_arg[0]["cache_control"] == {"type": "ephemeral"}


def test_anthropic_caching_on_skips_short_system(fake_anthropic):
    from hrag.providers.llm import AnthropicProvider

    cfg = LLMConfig(provider="anthropic", model="claude-3-5-sonnet-20241022",
                    anthropic_prompt_caching=True)
    p = AnthropicProvider(cfg)
    p.generate(_request_with("short system", "y" * 2048))

    sys_arg = fake_anthropic.kwargs["system"]
    # Below the 1024-char threshold → plain string fallback (no cache marker).
    assert isinstance(sys_arg, str)
    assert sys_arg == "short system"


def test_anthropic_caching_marks_last_user_message(fake_anthropic):
    from hrag.providers.llm import AnthropicProvider

    cfg = LLMConfig(provider="anthropic", model="claude-3-5-sonnet-20241022",
                    anthropic_prompt_caching=True)
    p = AnthropicProvider(cfg)
    big_user = "U" * 2048
    p.generate(_request_with("S" * 2048, big_user))

    last_user = fake_anthropic.kwargs["messages"][-1]
    assert last_user["role"] == "user"
    content = last_user["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert content[0]["text"] == big_user
    assert content[0]["cache_control"] == {"type": "ephemeral"}


def test_anthropic_caching_skips_short_user_message(fake_anthropic):
    from hrag.providers.llm import AnthropicProvider

    cfg = LLMConfig(provider="anthropic", model="claude-3-5-sonnet-20241022",
                    anthropic_prompt_caching=True)
    p = AnthropicProvider(cfg)
    p.generate(_request_with("S" * 2048, "short user"))

    last_user = fake_anthropic.kwargs["messages"][-1]
    # Below threshold → plain string (no cache wrap).
    assert isinstance(last_user["content"], str)
    assert last_user["content"] == "short user"


def test_anthropic_caching_streaming_path(fake_anthropic):
    from hrag.providers.llm import AnthropicProvider

    cfg = LLMConfig(provider="anthropic", model="claude-3-5-sonnet-20241022",
                    anthropic_prompt_caching=True)
    p = AnthropicProvider(cfg)
    big_system = "S" * 2048
    big_user = "U" * 2048
    # Consume the stream so the call actually happens.
    list(p.generate_stream(_request_with(big_system, big_user)))

    sys_arg = fake_anthropic.kwargs["system"]
    assert isinstance(sys_arg, list)
    assert sys_arg[0]["cache_control"] == {"type": "ephemeral"}

    last_user = fake_anthropic.kwargs["messages"][-1]
    assert isinstance(last_user["content"], list)
    assert last_user["content"][0]["cache_control"] == {"type": "ephemeral"}


def test_anthropic_caching_marks_only_last_user(fake_anthropic):
    """When several user messages exist, only the LAST one gets cache_control."""
    from hrag.providers.llm import AnthropicProvider

    cfg = LLMConfig(provider="anthropic", model="claude-3-5-sonnet-20241022",
                    anthropic_prompt_caching=True)
    p = AnthropicProvider(cfg)
    msgs = [
        Message(role="system", content="S" * 2048),
        Message(role="user", content="A" * 2048),
        Message(role="assistant", content="reply"),
        Message(role="user", content="B" * 2048),
    ]
    p.generate(GenerationRequest(messages=msgs, temperature=0.0, max_tokens=32))

    sent = fake_anthropic.kwargs["messages"]
    # First user message stays a string (not the suffix we want to cache).
    assert isinstance(sent[0]["content"], str)
    # Last user message is wrapped.
    assert isinstance(sent[-1]["content"], list)
    assert sent[-1]["content"][0]["text"] == "B" * 2048
