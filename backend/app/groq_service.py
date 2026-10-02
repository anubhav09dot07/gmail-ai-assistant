from __future__ import annotations

from typing import Any

from . import config


class GroqServiceError(RuntimeError):
    """A safe, non-secret error raised by the isolated Groq boundary."""


class GroqService:
    """Small lazy Groq client boundary with no application-specific retrieval logic."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        client: Any = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else config.GROQ_API_KEY
        self.model = model or config.GROQ_MODEL
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            if not self.api_key:
                raise GroqServiceError("Groq API key is not configured")
            try:
                from groq import Groq

                self._client = Groq(
                    api_key=self.api_key, timeout=config.GROQ_TIMEOUT_SECONDS
                )
            except Exception as error:
                raise GroqServiceError("Groq client is unavailable") from error
        return self._client

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        if not system_prompt.strip() or not user_prompt.strip():
            raise GroqServiceError("Groq prompts must not be empty")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = response.choices[0].message.content
        except GroqServiceError:
            raise
        except Exception as error:
            raise GroqServiceError("Groq request failed") from error
        if not isinstance(content, str) or not content.strip():
            raise GroqServiceError("Groq returned an empty response")
        return content.strip()