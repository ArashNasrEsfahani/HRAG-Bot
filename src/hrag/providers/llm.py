"""LLMProvider interface and concrete implementations.

Concrete adapters live in this same module for now. They are intentionally lightweight:
each implements a single `generate(messages, **kwargs) -> GenerationResponse` call.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Iterator, Optional

from hrag.config import LLMConfig
from hrag.types import GenerationRequest, GenerationResponse, Message


class LLMProviderError(RuntimeError):
    """LLM-provider-side error with a humane, actionable message."""


def _translate_ollama_error(exc: BaseException, model_name: str) -> LLMProviderError:
    """Convert a low-level ollama/http exception into a friendly LLMProviderError.

    Duck-types the exception rather than isinstance-checking the real ollama
    classes so that stub/mock objects work equally well in tests.
    """
    # Check for ResponseError-like objects (have status_code + error attributes).
    status_code = getattr(exc, "status_code", None)
    error_text = str(getattr(exc, "error", "")) or str(exc)
    exc_str = str(exc).lower()

    # 503 — server unavailable / out of memory / mid-upgrade
    if status_code == 503:
        return LLMProviderError(
            "Ollama is unavailable (503). The server may be starting, upgrading, or out of "
            "memory. Try again in a moment, or run `ollama serve` / restart Ollama."
        )

    # 404 or "model not found" messages
    model_keywords = ("model not found", "no such model", "pull model")
    if status_code == 404 or any(kw in error_text.lower() for kw in model_keywords) or any(
        kw in exc_str for kw in model_keywords
    ):
        return LLMProviderError(
            f"Ollama model {model_name!r} is not pulled. Run: ollama pull {model_name}"
        )

    # Connection refused / cannot reach Ollama
    conn_keywords = ("connection refused", "failed to establish a new connection")
    if any(kw in exc_str for kw in conn_keywords):
        return LLMProviderError(
            "Cannot reach Ollama at the configured base_url. "
            "Is `ollama serve` running? See https://ollama.com."
        )

    # Fallback for any other ollama/http error
    return LLMProviderError(f"Ollama call failed: {exc}")


class LLMProvider(ABC):
    """Abstract LLM backend. Sync API; streaming can be added later."""

    name: str = "abstract"

    def __init__(self, config: LLMConfig):
        self.config = config

    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Run inference, return the assistant text."""

    def generate_stream(self, request: GenerationRequest) -> Iterator[str]:
        """Stream tokens as they arrive. Default impl falls back to non-streaming."""
        yield self.generate(request).text

    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Convenience wrapper for one-shot prompting."""
        msgs: list[Message] = []
        if system:
            msgs.append(Message(role="system", content=system))
        msgs.append(Message(role="user", content=prompt))
        resp = self.generate(
            GenerationRequest(messages=msgs, temperature=temperature, max_tokens=max_tokens)
        )
        return resp.text

    def verify_ready(self) -> None:
        """Confirm the provider is reachable and the configured model is usable.

        Default impl: best-effort one-token completion. Subclasses may override
        with a faster/cheaper check. Must raise `LLMProviderError` on failure.
        """
        try:
            _ = self.complete("ping", max_tokens=1, temperature=0.0)
        except LLMProviderError:
            raise
        except Exception as exc:
            raise LLMProviderError(f"Provider readiness check failed: {exc}") from exc


def get_llm_provider(config: LLMConfig) -> LLMProvider:
    """Factory: pick an adapter based on `config.provider`."""
    name = config.provider.lower().strip()
    if name == "ollama":
        return OllamaProvider(config)
    if name == "openai":
        return OpenAIProvider(config)
    if name == "anthropic":
        return AnthropicProvider(config)
    raise ValueError(f"Unknown LLM provider: {config.provider!r}")


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            import ollama
        except ImportError as e:
            raise ImportError(
                "The 'ollama' package is required for OllamaProvider. "
                "Install with: pip install ollama"
            ) from e
        self._client = ollama.Client(host=config.base_url) if config.base_url else ollama.Client()

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        msgs = [{"role": m.role, "content": m.content} for m in request.messages]
        options = self._build_options(request)
        chat_kwargs = self._build_chat_kwargs(msgs, options)
        try:
            resp = self._client.chat(**chat_kwargs)
        except Exception as exc:
            raise _translate_ollama_error(exc, self.config.model) from exc
        text = resp["message"]["content"]
        return GenerationResponse(text=text, raw=resp)

    def generate_stream(self, request: GenerationRequest) -> Iterator[str]:
        msgs = [{"role": m.role, "content": m.content} for m in request.messages]
        options = self._build_options(request)
        chat_kwargs = self._build_chat_kwargs(msgs, options)
        chat_kwargs["stream"] = True
        try:
            stream = self._client.chat(**chat_kwargs)
            for chunk in stream:
                piece = chunk.get("message", {}).get("content", "")
                if piece:
                    yield piece
        except LLMProviderError:
            raise
        except Exception as exc:
            raise _translate_ollama_error(exc, self.config.model) from exc

    def _build_chat_kwargs(self, msgs: list[dict], options: dict) -> dict:
        kwargs: dict = {
            "model": self.config.model,
            "messages": msgs,
            "options": options,
        }
        think = getattr(self.config, "think", None)
        if think is not None:
            kwargs["think"] = think
        keep_alive = getattr(self.config, "keep_alive", None)
        if keep_alive is not None:
            kwargs["keep_alive"] = keep_alive
        return kwargs

    def _build_options(self, request: GenerationRequest) -> dict:
        options = {
            "temperature": request.temperature
            if request.temperature is not None
            else self.config.temperature,
            "num_predict": request.max_tokens
            if request.max_tokens is not None
            else self.config.max_tokens,
        }
        num_ctx = getattr(self.config, "num_ctx", None)
        if num_ctx is not None:
            options["num_ctx"] = num_ctx
        num_keep = getattr(self.config, "num_keep", None)
        if num_keep is not None and num_keep > 0:
            options["num_keep"] = int(num_keep)
        if request.stop:
            options["stop"] = request.stop
        return options

    def verify_ready(self) -> None:
        """Check Ollama reachability and model availability without loading the model.

        Hits /api/tags to list installed models (fast, no generation). Raises
        `LLMProviderError` with an actionable message if unreachable or the
        configured model is not present.
        """
        try:
            response = self._client.list()
        except LLMProviderError:
            raise
        except Exception as exc:
            exc_str = str(exc).lower()
            conn_keywords = ("connection refused", "failed to establish a new connection")
            if any(kw in exc_str for kw in conn_keywords):
                host = self.config.base_url or "http://localhost:11434"
                raise LLMProviderError(
                    f"Cannot reach Ollama at {host}. "
                    "Is `ollama serve` running? See https://ollama.com."
                ) from exc
            raise _translate_ollama_error(exc, self.config.model) from exc

        # The ollama client returns either an object with a `.models` attribute
        # or a dict with a "models" key — handle both.
        try:
            models_raw = response.models  # type: ignore[union-attr]
        except AttributeError:
            models_raw = response.get("models", []) if isinstance(response, dict) else []

        # Each entry is either an object with `.model` or a dict with "model".
        installed: list[str] = []
        for m in models_raw:
            try:
                name = m.model  # type: ignore[union-attr]
            except AttributeError:
                name = m.get("model", "") if isinstance(m, dict) else str(m)
            if name:
                # Strip ":latest" suffix for comparison so "llama3" matches "llama3:latest".
                installed.append(name)

        wanted = self.config.model
        # Accept exact match OR base-name match (strip ":latest").
        def _base(n: str) -> str:
            return n.split(":")[0]

        if not any(n == wanted or _base(n) == _base(wanted) for n in installed):
            available = ", ".join(installed) if installed else "(none)"
            raise LLMProviderError(
                f"Ollama model {wanted!r} is not pulled. "
                f"Available: {available}. "
                f"To pull it, run: ollama pull {wanted}"
            )


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        api_key = config.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OpenAI API key not found. Set OPENAI_API_KEY or config.llm.api_key."
            )
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "The 'openai' package is required for OpenAIProvider."
            ) from e
        self._client = OpenAI(api_key=api_key, base_url=config.base_url)

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        kwargs = self._build_kwargs(request)
        resp = self._client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content or ""
        return GenerationResponse(text=text, raw=resp)

    def generate_stream(self, request: GenerationRequest) -> Iterator[str]:
        kwargs = self._build_kwargs(request)
        kwargs["stream"] = True
        stream = self._client.chat.completions.create(**kwargs)
        for chunk in stream:
            choices = getattr(chunk, "choices", None)
            if not choices:
                continue
            delta = choices[0].delta
            piece = getattr(delta, "content", None)
            if piece:
                yield piece

    def _build_kwargs(self, request: GenerationRequest) -> dict:
        msgs = [{"role": m.role, "content": m.content} for m in request.messages]
        kwargs: dict = {
            "model": self.config.model,
            "messages": msgs,
            "temperature": request.temperature
            if request.temperature is not None
            else self.config.temperature,
            "max_tokens": request.max_tokens
            if request.max_tokens is not None
            else self.config.max_tokens,
        }
        if request.stop:
            kwargs["stop"] = request.stop
        return kwargs


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        api_key = config.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY or config.llm.api_key."
            )
        try:
            from anthropic import Anthropic
        except ImportError as e:
            raise ImportError(
                "The 'anthropic' package is required for AnthropicProvider."
            ) from e
        self._client = Anthropic(api_key=api_key, base_url=config.base_url)

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        kwargs = self._build_kwargs(request)
        resp = self._client.messages.create(**kwargs)
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        return GenerationResponse(text=text, raw=resp)

    def generate_stream(self, request: GenerationRequest) -> Iterator[str]:
        kwargs = self._build_kwargs(request)
        with self._client.messages.stream(**kwargs) as stream:
            for piece in stream.text_stream:
                if piece:
                    yield piece

    def _build_kwargs(self, request: GenerationRequest) -> dict:
        # Anthropic API requires `system` to be top-level, not a message role.
        system_text = ""
        msgs = []
        for m in request.messages:
            if m.role == "system":
                system_text = (system_text + "\n\n" + m.content).strip()
            else:
                msgs.append({"role": m.role, "content": m.content})
        kwargs: dict = {
            "model": self.config.model,
            "messages": msgs,
            "temperature": request.temperature
            if request.temperature is not None
            else self.config.temperature,
            "max_tokens": request.max_tokens
            if request.max_tokens is not None
            else self.config.max_tokens,
        }
        if system_text:
            kwargs["system"] = system_text
        if request.stop:
            kwargs["stop_sequences"] = request.stop
        return kwargs
