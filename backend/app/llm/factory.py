from __future__ import annotations

from typing import Any
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq

from .. import config


def get_chat_model(
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.0,
    api_key: str | None = None,
    timeout: float | None = None,
    **kwargs: Any,
) -> BaseChatModel:
    """Factory creating a standard LangChain ChatModel according to configuration.

    Defaults to Groq with openai/gpt-oss-120b.
    """
    selected_provider = (provider or config.LLM_PROVIDER).lower()
    selected_model = model or config.LLM_MODEL
    selected_timeout = timeout if timeout is not None else config.GROQ_TIMEOUT_SECONDS

    if selected_provider == "groq":
        key = api_key or config.GROQ_API_KEY
        return ChatGroq(
            model=selected_model,
            groq_api_key=key,
            temperature=temperature,
            timeout=selected_timeout,
            **kwargs,
        )

    raise ValueError(f"Unsupported LLM provider: {selected_provider!r}")
