"""LLM providers behind one small interface, with retries, usage accounting and caching."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import threading
import time
from dataclasses import dataclass

from . import prompts

log = logging.getLogger(__name__)

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}
RETRYABLE_NAMES = ("RateLimit", "Timeout", "Connection", "Overloaded", "InternalServer",
                   "ServiceUnavailable", "ResourceExhausted", "DeadlineExceeded")


class LLMError(RuntimeError):
    """A model call failed (after retries) or returned nothing usable."""


@dataclass
class Completion:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0


class Provider:
    name = "base"
    model = ""

    def complete(self, prompt: str, system: str | None) -> Completion:
        raise NotImplementedError


class OpenAIProvider(Provider):
    name = "openai"

    def __init__(self, model=None, api_key=None, base_url=None):
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._base_url = base_url or os.getenv("OPENAI_BASE_URL") or None
        self._client = None
        self._lock = threading.Lock()

    def _get_client(self):
        with self._lock:
            if self._client is None:
                from openai import OpenAI

                if not self._api_key:
                    raise LLMError(f"{self.name}: set OPENAI_API_KEY (or the provider's key variable)")
                self._client = OpenAI(api_key=self._api_key, base_url=self._base_url)
            return self._client

    def complete(self, prompt, system):
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}]
        response = self._get_client().chat.completions.create(model=self.model, messages=messages)
        usage = response.usage
        return Completion(
            response.choices[0].message.content or "",
            getattr(usage, "prompt_tokens", 0) or 0,
            getattr(usage, "completion_tokens", 0) or 0,
        )


class VLLMProvider(Provider):
    """An open model served by vLLM (or any OpenAI-compatible server: SGLang, Ollama, ...).

    ``VLLM_API_BASE`` points at the server (default ``http://localhost:8000/v1``). When
    ``VLLM_MODEL`` is not set, the first model the server lists is used. The server's
    ``max_model_len`` is read too, so prompts can be sized to fit the context window.
    """

    name = "vllm"

    def __init__(self, model=None, base_url=None):
        self.base_url = (base_url or os.getenv("VLLM_API_BASE", "http://localhost:8000/v1")).rstrip("/")
        self._model = model or os.getenv("VLLM_MODEL") or None
        self._api_key = os.getenv("VLLM_API_KEY", "EMPTY")
        self.timeout = float(os.getenv("VLLM_TIMEOUT", "600"))
        self.max_tokens = int(os.getenv("VLLM_MAX_TOKENS", "0")) or None
        temperature = os.getenv("VLLM_TEMPERATURE")
        self.temperature = float(temperature) if temperature else None
        self._context_length = int(os.getenv("VLLM_MAX_MODEL_LEN", "0")) or None
        self._client = None
        self._discovered = False
        self._lock = threading.Lock()
        self._discover_lock = threading.Lock()

    def _get_client(self):
        with self._lock:
            if self._client is None:
                from openai import OpenAI

                self._client = OpenAI(api_key=self._api_key, base_url=self.base_url,
                                      timeout=self.timeout, max_retries=0)
            return self._client

    def _discover(self) -> None:
        """Ask the server which model it serves and how long its context is."""
        with self._discover_lock:
            if not self._discovered:
                self._discover_locked()
                self._discovered = True

    def _discover_locked(self) -> None:
        try:
            listing = self._get_client().models.list()
            served = list(listing.data)
        except Exception as exc:
            if self._model is None:
                raise LLMError(f"vllm: cannot list models at {self.base_url} ({exc}); is the server "
                               "running? Set VLLM_API_BASE / VLLM_MODEL.") from exc
            return
        if not served:
            if self._model is None:
                raise LLMError(f"vllm: the server at {self.base_url} serves no models")
            return
        chosen = next((m for m in served if m.id == self._model), None)
        if self._model is None:
            chosen = served[0]
            self._model = chosen.id
            log.info("vllm: using served model %s", self._model)
        if chosen is not None and self._context_length is None:
            length = getattr(chosen, "max_model_len", None) or (chosen.model_extra or {}).get("max_model_len")
            self._context_length = int(length) if length else None

    @property
    def model(self) -> str:
        if self._model is None:
            self._discover()
        return self._model or ""

    @property
    def context_length(self) -> int | None:
        self._discover()
        return self._context_length

    def complete(self, prompt, system):
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}]
        kwargs = {}
        if self.max_tokens:
            kwargs["max_tokens"] = self.max_tokens
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        response = self._get_client().chat.completions.create(model=self.model, messages=messages, **kwargs)
        usage = response.usage
        choice = response.choices[0]
        if choice.finish_reason == "length":
            log.warning("vllm: answer was cut off at the token limit")
        return Completion(
            choice.message.content or "",
            getattr(usage, "prompt_tokens", 0) or 0,
            getattr(usage, "completion_tokens", 0) or 0,
        )


def strip_reasoning(text: str) -> str:
    """Drop ``<think>...</think>`` blocks that reasoning models (Qwen3, DeepSeek-R1) emit."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    if "</think>" in text:  # opening tag consumed by the chat template
        text = text.split("</think>", 1)[1]
    return text.strip()


class AnthropicProvider(Provider):
    """Claude through the Anthropic API, or through Google Vertex AI when a project is set."""

    name = "anthropic"

    def __init__(self, model=None):
        self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
        self.max_tokens = int(os.getenv("ANTHROPIC_MAX_TOKENS", "8192"))
        self._client = None
        self._lock = threading.Lock()

    def _get_client(self):
        with self._lock:
            if self._client is None:
                project = os.getenv("ANTHROPIC_VERTEX_PROJECT_ID") or os.getenv("PROJECT_ID")
                if project:
                    from anthropic import AnthropicVertex

                    region = os.getenv("VERTEX_LOCATION") or os.getenv("CLOUD_ML_REGION") or "us-east5"
                    self._client = AnthropicVertex(region=region, project_id=project)
                else:
                    from anthropic import Anthropic

                    if not os.getenv("ANTHROPIC_API_KEY"):
                        raise LLMError("anthropic: set ANTHROPIC_API_KEY, or PROJECT_ID for Vertex AI")
                    self._client = Anthropic()
            return self._client

    def complete(self, prompt, system):
        kwargs = {"system": system} if system else {}
        message = self._get_client().messages.create(
            model=self.model, max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}], **kwargs)
        text = "".join(block.text for block in message.content if getattr(block, "type", "") == "text")
        return Completion(text, message.usage.input_tokens, message.usage.output_tokens)


class GeminiProvider(Provider):
    name = "gemini"

    def __init__(self, model=None):
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        self._client = None
        self._lock = threading.Lock()

    def _get_client(self):
        with self._lock:
            if self._client is None:
                from google import genai

                key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
                if not key:
                    raise LLMError("gemini: set GEMINI_API_KEY")
                self._client = genai.Client(api_key=key)
            return self._client

    def complete(self, prompt, system):
        config = None
        if system:
            from google.genai import types

            config = types.GenerateContentConfig(system_instruction=system)
        response = self._get_client().models.generate_content(
            model=self.model, contents=prompt, config=config)
        usage = response.usage_metadata
        return Completion(
            response.text or "",
            getattr(usage, "prompt_token_count", 0) or 0,
            getattr(usage, "candidates_token_count", 0) or 0,
        )


class MockProvider(Provider):
    """Deterministic offline provider for tests and dry runs (no network, no key)."""

    name = "mock"
    model = "mock"

    def complete(self, prompt, system):
        digest = hashlib.sha256(prompt.encode()).hexdigest()[:8]
        if prompts.JUDGE_MARKER in prompt:
            text = json.dumps({"best": "A"})
        elif prompts.MCQ_MARKER in prompt:
            text = "A"
        elif prompts.BASELINE_MARKER in prompt:
            ids = re.findall(r"^- `(.+)`$", prompt.split(prompts.BASELINE_MARKER, 1)[1], re.M)
            text = json.dumps({uid: f"Mock summary of {uid}." for uid in ids})
        else:
            subject = re.search(r"^# Code: .*?`(.+?)`", prompt, re.M)
            text = f"Mock summary of {subject.group(1) if subject else 'input'} ({digest})."
        return Completion(text, len(prompt) // 4, len(text) // 4)


# vLLM is the default for every task; the hosted APIs are optional extras.
DEFAULT_PROVIDER = os.getenv("ECB_PROVIDER", "vllm")
PROVIDERS = {
    "vllm": VLLMProvider,
    "vllm_models": VLLMProvider,  # name used by earlier versions
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
    "mock": MockProvider,
}
PROVIDER_CHOICES = ("vllm", "openai", "anthropic", "gemini", "mock")


def make_provider(name: str | None = None, model: str | None = None, base_url: str | None = None) -> Provider:
    name = name or DEFAULT_PROVIDER
    try:
        cls = PROVIDERS[name]
    except KeyError:
        raise ValueError(f"Unknown provider {name!r}; choose from {', '.join(PROVIDER_CHOICES)}") from None
    if cls is MockProvider:
        return cls()
    if cls is VLLMProvider:
        return cls(model=model, base_url=base_url)
    if base_url and cls is OpenAIProvider:
        return cls(model=model, base_url=base_url)
    return cls(model=model)


def default_label(provider: Provider) -> str:
    """Name used in output files: the model for open models, else the provider."""
    if isinstance(provider, VLLMProvider):
        short = provider.model.rsplit("/", 1)[-1]
        return re.sub(r"[^A-Za-z0-9]+", "_", short).strip("_").lower() or provider.name
    return provider.name


def _retryable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if isinstance(status, int) and status in RETRYABLE_STATUS:
        return True
    return any(token in type(exc).__name__ for token in RETRYABLE_NAMES)


class LLM:
    """Thread-safe wrapper adding retries, token accounting and an optional disk cache."""

    def __init__(self, provider: Provider, cache_dir: str | None = None, max_retries: int = 5,
                 system: str | None = prompts.SYSTEM):
        self.provider = provider
        self.cache_dir = cache_dir
        self.max_retries = max_retries
        self.system = system
        self._lock = threading.Lock()
        self.usage = {"calls": 0, "cached_calls": 0, "input_tokens": 0, "output_tokens": 0}

    @property
    def name(self) -> str:
        return self.provider.name

    @property
    def model(self) -> str:
        return self.provider.model

    def _cache_path(self, prompt: str, tag: str) -> str | None:
        if not self.cache_dir:
            return None
        key = hashlib.sha256(json.dumps(
            [self.provider.name, self.provider.model, self.system, prompt, tag]).encode()).hexdigest()
        return os.path.join(self.cache_dir, key[:2], f"{key}.json")

    def complete(self, prompt: str, *, cache_tag: str = "") -> str:
        path = self._cache_path(prompt, cache_tag)
        if path and os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as handle:
                    text = json.load(handle)["text"]
                with self._lock:
                    self.usage["cached_calls"] += 1
                return text
            except (OSError, ValueError, KeyError):
                pass

        delay = 1.0
        for attempt in range(self.max_retries + 1):
            try:
                result = self.provider.complete(prompt, self.system)
                break
            except LLMError:
                raise
            except Exception as exc:  # provider SDK errors
                if attempt >= self.max_retries or not _retryable(exc):
                    raise LLMError(f"{self.provider.name}: {type(exc).__name__}: {exc}") from exc
                wait = delay * (1 + random.random() * 0.25)
                log.warning("%s call failed (%s); retrying in %.1fs", self.provider.name,
                            type(exc).__name__, wait)
                time.sleep(wait)
                delay = min(delay * 2, 60)

        with self._lock:
            self.usage["calls"] += 1
            self.usage["input_tokens"] += result.input_tokens
            self.usage["output_tokens"] += result.output_tokens
        text = strip_reasoning(result.text or "")
        if not text:
            raise LLMError(f"{self.provider.name}: empty response")
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump({"text": text}, handle)
            os.replace(tmp, path)
        return text
