from __future__ import annotations

import logging
from typing import Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser

from ..llm.factory import get_chat_model
from ..vector_store import RetrievedChunk
from .citations import RAGSource
from .prompts import (
    NO_EVIDENCE_ANSWER,
    build_gmail_context,
    clean_generated_answer,
    get_grounding_prompt,
)

logger = logging.getLogger(__name__)


def structured_match_summary(sources: Sequence[RAGSource]) -> str:
    """Safe fallback when a model declines to summarize exact Gmail matches."""
    count = len(sources)
    noun = "email" if count == 1 else "emails"
    details = "; ".join(
        f"{source.subject or 'No subject'} from {source.sender or 'Unknown sender'}"
        + (f" ({source.date})" if source.date else "")
        for source in sources
    )
    return f"I found {count} matching {noun}: {details}."


def generate_grounded_answer(
    query: str,
    chunks: Sequence[RetrievedChunk],
    *,
    chat_model: BaseChatModel | None = None,
    is_structured_search: bool = False,
    sources: Sequence[RAGSource] = (),
) -> tuple[str | None, str]:
    """Execute LangChain grounding chain over verified email context."""
    context = build_gmail_context(chunks)
    if not context:
        return None, "insufficient_gmail_evidence"

    prompt = get_grounding_prompt()
    model = chat_model or get_chat_model(temperature=0.0)
    chain = prompt | model | StrOutputParser()

    try:
        raw_answer = chain.invoke({"query": query, "context": context})
        if not isinstance(raw_answer, str) or not raw_answer.strip():
            logger.warning("LLM returned empty answer")
            return None, "generation_unavailable"

        answer = clean_generated_answer(raw_answer)

        # If model declined to summarize an exact structured list query, use deterministic fallback
        if (
            answer == NO_EVIDENCE_ANSWER
            and is_structured_search
            and sources
        ):
            answer = structured_match_summary(sources)

        return answer, "grounded_answer_generated"
    except Exception as exc:
        logger.error("LangChain generation chain failed: %s", exc)
        return None, "generation_unavailable"
