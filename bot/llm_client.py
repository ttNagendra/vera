"""
llm_client.py — Multi-provider LLM adapter for Vera bot.

Supports: Gemini, OpenAI, Anthropic, Groq, DeepSeek.
All providers use temperature=0 for determinism.
Retry logic: 3 attempts with exponential backoff.
Timeout: 25s (leaves buffer for judge's 30s limit).
"""

import os
import json
import time
import httpx
import asyncio
from abc import ABC, abstractmethod
from typing import Optional


TIMEOUT = 25.0
MAX_RETRIES = 3


class LLMClient(ABC):
    """Abstract base for all LLM providers."""

    @abstractmethod
    async def complete(self, system: str, user: str) -> str:
        """Return the assistant's text response."""
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        pass


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

async def _with_retry(fn, max_retries=MAX_RETRIES):
    last_err = None
    for attempt in range(max_retries):
        try:
            return await fn()
        except (httpx.TimeoutException, httpx.ConnectError) as e:
            last_err = e
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                # Rate limit — back off longer
                wait = 5 * (attempt + 1)  # 5s, 10s, 15s
                await asyncio.sleep(wait)
                last_err = e
            else:
                raise
    raise last_err


# ---------------------------------------------------------------------------
# Google Gemini
# ---------------------------------------------------------------------------

class GeminiClient(LLMClient):
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        self.api_key = api_key
        self.model = model
        self._base = "https://generativelanguage.googleapis.com/v1beta/models"

    @property
    def provider_name(self) -> str:
        return f"gemini/{self.model}"

    async def complete(self, system: str, user: str) -> str:
        url = f"{self._base}/{self.model}:generateContent?key={self.api_key}"
        # Gemini uses a system_instruction separate from user content
        body = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"parts": [{"text": user}], "role": "user"}],
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": 1500,
                "responseMimeType": "text/plain",
            },
        }

        async def call():
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                resp = await client.post(url, json=body)
                resp.raise_for_status()
                data = resp.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]

        return await _with_retry(call)


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

class OpenAIClient(LLMClient):
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.model = model

    @property
    def provider_name(self) -> str:
        return f"openai/{self.model}"

    async def complete(self, system: str, user: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.0,
            "max_tokens": 1500,
        }

        async def call():
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                resp = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers, json=body
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]

        return await _with_retry(call)


# ---------------------------------------------------------------------------
# Anthropic Claude
# ---------------------------------------------------------------------------

class AnthropicClient(LLMClient):
    def __init__(self, api_key: str, model: str = "claude-3-5-haiku-20241022"):
        self.api_key = api_key
        self.model = model

    @property
    def provider_name(self) -> str:
        return f"anthropic/{self.model}"

    async def complete(self, system: str, user: str) -> str:
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        body = {
            "model": self.model,
            "max_tokens": 1500,
            "temperature": 0.0,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }

        async def call():
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                resp = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers, json=body
                )
                resp.raise_for_status()
                return resp.json()["content"][0]["text"]

        return await _with_retry(call)


# ---------------------------------------------------------------------------
# Groq
# ---------------------------------------------------------------------------

class GroqClient(LLMClient):
    def __init__(self, api_key: str, model: str = "llama-3.1-70b-versatile"):
        self.api_key = api_key
        self.model = model

    @property
    def provider_name(self) -> str:
        return f"groq/{self.model}"

    async def complete(self, system: str, user: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.0,
            "max_tokens": 1500,
        }

        async def call():
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                resp = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers, json=body
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]

        return await _with_retry(call)


# ---------------------------------------------------------------------------
# DeepSeek
# ---------------------------------------------------------------------------

class DeepSeekClient(LLMClient):
    def __init__(self, api_key: str, model: str = "deepseek-chat"):
        self.api_key = api_key
        self.model = model

    @property
    def provider_name(self) -> str:
        return f"deepseek/{self.model}"

    async def complete(self, system: str, user: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.0,
            "max_tokens": 1500,
        }

        async def call():
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                resp = await client.post(
                    "https://api.deepseek.com/v1/chat/completions",
                    headers=headers, json=body
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]

        return await _with_retry(call)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

DEFAULTS = {
    "gemini":    ("GeminiClient",    "gemini-2.5-flash"),
    "openai":    ("OpenAIClient",    "gpt-4o-mini"),
    "anthropic": ("AnthropicClient", "claude-3-5-haiku-20241022"),
    "groq":      ("GroqClient",      "llama-3.1-70b-versatile"),
    "deepseek":  ("DeepSeekClient",  "deepseek-chat"),
}

CLIENTS = {
    "gemini":    GeminiClient,
    "openai":    OpenAIClient,
    "anthropic": AnthropicClient,
    "groq":      GroqClient,
    "deepseek":  DeepSeekClient,
}


def create_llm_client() -> LLMClient:
    """Create LLM client from environment variables."""
    provider = os.getenv("LLM_PROVIDER", "gemini").lower().strip()
    api_key = os.getenv("LLM_API_KEY", "")
    model = os.getenv("LLM_MODEL", "").strip()

    if provider not in CLIENTS:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{provider}'. "
            f"Valid options: {', '.join(CLIENTS.keys())}"
        )

    _, default_model = DEFAULTS[provider]
    chosen_model = model or default_model

    cls = CLIENTS[provider]
    if provider == "groq":
        # Groq doesn't need api_key validation the same way
        return cls(api_key or "", chosen_model)
    return cls(api_key, chosen_model)
