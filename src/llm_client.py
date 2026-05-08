from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from openai import OpenAI

from .config import Settings


@dataclass
class LLMClient:
    settings: Settings
    logger: logging.Logger

    def __post_init__(self) -> None:
        self.enabled = bool(self.settings.openai_api_key)
        self.failed_permanently = False
        client_kwargs = {
            "api_key": self.settings.openai_api_key,
            "timeout": self.settings.openai_timeout_seconds,
        }
        if self.settings.openai_base_url:
            client_kwargs["base_url"] = self.settings.openai_base_url
        self.client = OpenAI(**client_kwargs) if self.enabled else None
        if not self.enabled:
            self.logger.warning("OPENAI_API_KEY is not set. LLM features will use fallback behavior.")
        elif self.settings.openai_base_url:
            self.logger.info("Using OpenAI-compatible API base URL: %s", self.settings.openai_base_url)

    def available(self) -> bool:
        return self.enabled and self.client is not None

    def health_check(self) -> bool:
        if not self.available():
            return False
        result = self.chat("你是连通性测试助手。", "只回复 OK", temperature=0)
        return bool(result.strip())

    def chat(self, system_prompt: str, user_prompt: str, temperature: float = 0.1) -> str:
        if not self.available():
            return ""
        try:
            return self._chat_with_retry(system_prompt, user_prompt, temperature)
        except Exception as exc:
            self.logger.exception("LLM call failed; this request will use fallback: %s", exc)
            return ""

    def _chat_with_retry(self, system_prompt: str, user_prompt: str, temperature: float) -> str:
        assert self.client is not None
        last_exc: Exception | None = None
        attempts = max(1, self.settings.openai_max_retries)
        for attempt in range(1, attempts + 1):
            try:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                if self.settings.openai_stream:
                    return self._chat_stream(messages, temperature)
                return self._chat_once(messages, temperature)
            except Exception as exc:
                last_exc = exc
                self.logger.warning("LLM attempt %s/%s failed: %s", attempt, attempts, exc)
                if attempt < attempts:
                    time.sleep(min(2 * attempt, 6))
        assert last_exc is not None
        raise last_exc

    def _chat_once(self, messages: list[dict[str, str]], temperature: float) -> str:
        assert self.client is not None
        response = self.client.chat.completions.create(
            model=self.settings.openai_model,
            messages=messages,
            temperature=temperature,
            max_tokens=self.settings.openai_max_output_tokens,
        )
        return response.choices[0].message.content or ""

    def _chat_stream(self, messages: list[dict[str, str]], temperature: float) -> str:
        assert self.client is not None
        stream = self.client.chat.completions.create(
            model=self.settings.openai_model,
            messages=messages,
            temperature=temperature,
            max_tokens=self.settings.openai_max_output_tokens,
            stream=True,
        )
        parts: list[str] = []
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)
            if content:
                parts.append(content)
        return "".join(parts)
