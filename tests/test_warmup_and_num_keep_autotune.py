"""Phase 9.4 — Ollama warm-up ping + num_keep auto-tune.

Tests verify:
  - Default flag values (warmup_on_init=True, num_keep_auto=False).
  - OllamaProvider.warmup() signature and behaviour.
  - estimate_num_keep() pure function.
  - Orchestrator._maybe_warmup_llm() routing logic (without a full Orchestrator).
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# 1. Config defaults
# ---------------------------------------------------------------------------


def test_warmup_default_on():
    from hrag.config import LLMConfig

    cfg = LLMConfig()
    assert cfg.warmup_on_init is True


def test_num_keep_auto_default_off():
    from hrag.config import LLMConfig

    cfg = LLMConfig()
    assert cfg.num_keep_auto is False


# ---------------------------------------------------------------------------
# 2. OllamaProvider.warmup() — signature
# ---------------------------------------------------------------------------


def test_warmup_method_signature():
    from hrag.providers.llm import OllamaProvider

    sig = inspect.signature(OllamaProvider.warmup)
    params = list(sig.parameters.keys())
    # Only 'self' — no extra required args.
    assert params == ["self"], f"Unexpected extra params: {params}"


# ---------------------------------------------------------------------------
# 3. warmup() calls _client.chat with num_predict=1
# ---------------------------------------------------------------------------


def test_warmup_calls_chat_with_one_token():
    """warmup() must call _client.chat with options['num_predict']==1."""
    from hrag.providers.llm import OllamaProvider
    from hrag.config import LLMConfig

    cfg = LLMConfig(provider="ollama", model="test-model", num_ctx=None, keep_alive=None)
    provider = object.__new__(OllamaProvider)
    provider.config = cfg
    provider._client = MagicMock()
    # Patch _build_chat_kwargs to intercept the options dict
    captured = {}

    original_build = OllamaProvider._build_chat_kwargs

    def spy_build(self, msgs, options):
        captured["options"] = dict(options)
        return original_build(self, msgs, options)

    with patch.object(OllamaProvider, "_build_chat_kwargs", spy_build):
        provider.warmup()

    assert "num_predict" in captured["options"], "num_predict must be in options"
    assert captured["options"]["num_predict"] == 1


# ---------------------------------------------------------------------------
# 4. warmup() uses keep_alive as a TOP-LEVEL kwarg (contract 14)
# ---------------------------------------------------------------------------


def test_warmup_uses_keep_alive_from_config():
    """keep_alive must appear as a top-level kwarg in the call to _client.chat
    (Phase 6 contract 14 — not inside options)."""
    from hrag.providers.llm import OllamaProvider
    from hrag.config import LLMConfig

    cfg = LLMConfig(provider="ollama", model="test-model", keep_alive="5m", num_ctx=None)
    provider = object.__new__(OllamaProvider)
    provider.config = cfg
    mock_client = MagicMock()
    provider._client = mock_client

    provider.warmup()

    # _client.chat should have been called once
    mock_client.chat.assert_called_once()
    call_kwargs = mock_client.chat.call_args[1] if mock_client.chat.call_args[1] else mock_client.chat.call_args[0][0] if mock_client.chat.call_args[0] else {}
    # _build_chat_kwargs builds a dict passed via **kwargs
    # .call_args is (args, kwargs); the dict is passed positionally as **kwargs
    all_kwargs = mock_client.chat.call_args.kwargs if hasattr(mock_client.chat.call_args, "kwargs") else {}
    if not all_kwargs and mock_client.chat.call_args:
        # Fallback: inspect args[0] if called as chat(**d)
        pass
    # The reliable check: _build_chat_kwargs returns a dict, then called as
    # self._client.chat(**chat_kwargs) — so keep_alive appears as a kwarg.
    assert "keep_alive" in all_kwargs, (
        f"keep_alive must be a top-level kwarg; found keys: {list(all_kwargs.keys())}"
    )
    assert all_kwargs["keep_alive"] == "5m"


# ---------------------------------------------------------------------------
# 5. warmup() puts num_keep inside options (contract 22)
# ---------------------------------------------------------------------------


def test_warmup_uses_num_keep_when_set():
    """num_keep must appear inside options, never at the top level."""
    from hrag.providers.llm import OllamaProvider
    from hrag.config import LLMConfig

    cfg = LLMConfig(provider="ollama", model="test-model", num_keep=128, num_ctx=None, keep_alive=None)
    provider = object.__new__(OllamaProvider)
    provider.config = cfg
    mock_client = MagicMock()
    provider._client = mock_client

    provider.warmup()

    mock_client.chat.assert_called_once()
    all_kwargs = mock_client.chat.call_args.kwargs
    # num_keep must be inside options, not at the top level
    assert "num_keep" not in all_kwargs, "num_keep must NOT be a top-level kwarg"
    assert all_kwargs.get("options", {}).get("num_keep") == 128, (
        "num_keep must be inside options"
    )


# ---------------------------------------------------------------------------
# 6. warmup() translates connection errors into LLMProviderError
# ---------------------------------------------------------------------------


def test_warmup_handles_connection_error():
    """A connection-refused exception must be re-raised as LLMProviderError."""
    from hrag.providers.llm import OllamaProvider, LLMProviderError
    from hrag.config import LLMConfig
    import pytest

    cfg = LLMConfig(provider="ollama", model="test-model", num_ctx=None, keep_alive=None)
    provider = object.__new__(OllamaProvider)
    provider.config = cfg
    mock_client = MagicMock()
    mock_client.chat.side_effect = ConnectionRefusedError("connection refused")
    provider._client = mock_client

    with pytest.raises(LLMProviderError):
        provider.warmup()


# ---------------------------------------------------------------------------
# 7. estimate_num_keep — pure function tests
# ---------------------------------------------------------------------------


def test_estimate_num_keep_pure_function():
    """400 chars → 100 tokens + 16 slack = 116."""
    from hrag.providers.llm import estimate_num_keep

    result = estimate_num_keep("a" * 400)
    assert result == 116  # 400 // 4 + 16


def test_estimate_num_keep_empty_string():
    """Empty string → 0 (not slack, because the guard returns early)."""
    from hrag.providers.llm import estimate_num_keep

    assert estimate_num_keep("") == 0


def test_estimate_num_keep_small_slack():
    """'hello' (5 chars) → max(1, 5//4) + 0 = 1 + 0 = 1."""
    from hrag.providers.llm import estimate_num_keep

    assert estimate_num_keep("hello", slack=0) == 1


# ---------------------------------------------------------------------------
# 8 & 9. Orchestrator._maybe_warmup_llm routing — without a full Orchestrator
# ---------------------------------------------------------------------------


def _make_fake_orchestrator(llm_name: str = "ollama", warmup_on_init: bool = True) -> SimpleNamespace:
    """Build a minimal namespace that satisfies _maybe_warmup_llm's attribute accesses."""
    mock_llm = MagicMock()
    mock_llm.name = llm_name
    mock_llm.warmup = MagicMock()

    from hrag.config import LLMConfig

    llm_cfg = LLMConfig(warmup_on_init=warmup_on_init, num_keep_auto=False)
    from hrag.config import Config

    cfg = Config(llm=llm_cfg)

    ns = SimpleNamespace(
        llm=mock_llm,
        config=cfg,
    )
    # Bind _maybe_warmup_llm to the namespace so it works like a method
    from hrag.orchestrator import Orchestrator

    ns._maybe_warmup_llm = lambda c: Orchestrator._maybe_warmup_llm(ns, c)
    return ns


def test_orchestrator_warmup_skipped_for_non_ollama():
    """warmup() must NOT be called when the provider name is not 'ollama'."""
    ns = _make_fake_orchestrator(llm_name="openai", warmup_on_init=True)
    ns._maybe_warmup_llm(ns.config)
    ns.llm.warmup.assert_not_called()


def test_orchestrator_warmup_skipped_when_flag_off():
    """warmup() must NOT be called when warmup_on_init=False even with Ollama."""
    ns = _make_fake_orchestrator(llm_name="ollama", warmup_on_init=False)
    ns._maybe_warmup_llm(ns.config)
    ns.llm.warmup.assert_not_called()


# ---------------------------------------------------------------------------
# 10. num_keep auto-tune via _maybe_warmup_llm
# ---------------------------------------------------------------------------


def test_num_keep_auto_tune_sets_value():
    """When num_keep_auto=True and num_keep is unset, auto-tune must populate num_keep."""
    from hrag.config import Config, LLMConfig
    from hrag.orchestrator import Orchestrator

    llm_cfg = LLMConfig(warmup_on_init=False, num_keep_auto=True, num_keep=None)
    cfg = Config(llm=llm_cfg)

    mock_llm = MagicMock()
    mock_llm.name = "ollama"

    ns = SimpleNamespace(llm=mock_llm, config=cfg)
    ns._maybe_warmup_llm = lambda c: Orchestrator._maybe_warmup_llm(ns, c)

    ns._maybe_warmup_llm(cfg)

    # num_keep should now be a positive integer
    assert cfg.llm.num_keep is not None
    assert isinstance(cfg.llm.num_keep, int)
    assert cfg.llm.num_keep > 0


def test_num_keep_auto_tune_does_not_override_manual():
    """When num_keep is already set, auto-tune must leave it alone."""
    from hrag.config import Config, LLMConfig
    from hrag.orchestrator import Orchestrator

    llm_cfg = LLMConfig(warmup_on_init=False, num_keep_auto=True, num_keep=42)
    cfg = Config(llm=llm_cfg)

    mock_llm = MagicMock()
    mock_llm.name = "ollama"

    ns = SimpleNamespace(llm=mock_llm, config=cfg)
    ns._maybe_warmup_llm = lambda c: Orchestrator._maybe_warmup_llm(ns, c)

    ns._maybe_warmup_llm(cfg)

    # Manual value must be preserved
    assert cfg.llm.num_keep == 42
