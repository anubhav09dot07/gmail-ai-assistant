from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from langchain_core.language_models.chat_models import BaseChatModel

from . import config
from .groq_service import GroqService, GroqServiceError
from .llm.factory import get_chat_model
from .query_service import GmailQueryEvaluation, evaluate_gmail_query
from .rag.chains import generate_grounded_answer, structured_match_summary
from .rag.citations import RAGSource, deduplicate_sources, gmail_url, source_from_chunk
from .rag.prompts import (
    GROUNDING_SYSTEM_PROMPT,
    NO_EVIDENCE_ANSWER,
    build_gmail_context,
    clean_generated_answer,
)
from .vector_store import RetrievedChunk

logger = logging.getLogger(__name__)


class GroqGenerator(Protocol):
    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        ...


@dataclass(frozen=True)
class RAGResult:
    query: str
    answer: str | None
    decision: str
    reason: str
    sources: tuple[RAGSource, ...] = ()
    evaluation: GmailQueryEvaluation | None = None

    def public_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "decision": self.decision,
            "reason": self.reason,
            "sources": [source.public_dict() for source in self.sources],
        }


def _user_prompt(query: str, context: str) -> str:
    return (
        "<user_question>\n"
        f"{query}\n"
        "</user_question>\n\n"
        "<retrieved_gmail_evidence>\n"
        f"{context}\n"
        "</retrieved_gmail_evidence>\n\n"
        "Answer only from the retrieved Gmail evidence."
    )


# Internal aliases for backward compatibility with existing tests
_gmail_url = gmail_url
_source = source_from_chunk
_deduplicate_sources = deduplicate_sources
_clean_generated_answer = clean_generated_answer
_structured_match_summary = structured_match_summary


def generate_gmail_answer(
    user_id: str,
    query: str,
    *,
    evaluator: Callable[..., GmailQueryEvaluation] = evaluate_gmail_query,
    groq_service: GroqGenerator | None = None,
    chat_model: BaseChatModel | None = None,
    gmail_service=None,
) -> RAGResult:
    """Evaluate evidence and generate grounded answer using LangChain or injected generator."""
    if gmail_service is None:
        try:
            from .gmail import _gmail_service
            gmail_service = _gmail_service()
        except Exception:
            gmail_service = None

    import inspect
    sig = inspect.signature(evaluator)
    if "gmail_service" in sig.parameters or any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    ):
        evaluation = evaluator(user_id, query, gmail_service=gmail_service)
    else:
        evaluation = evaluator(user_id, query)

    if evaluation.decision != "allow":
        if config.RAG_DEBUG:
            logger.info("Query rejected before generation: %s", evaluation.reason)
        return RAGResult(
            query=evaluation.query,
            answer=None,
            decision="reject",
            reason=evaluation.reason,
            evaluation=evaluation,
        )

    context = build_gmail_context(evaluation.retrieved_chunks)
    if not context:
        return RAGResult(
            query=evaluation.query,
            answer=None,
            decision="reject",
            reason="insufficient_gmail_evidence",
            evaluation=evaluation,
        )

    sources = deduplicate_sources(evaluation.retrieved_chunks)
    plan = evaluation.search_plan

    # 1. Custom/mock generator path (used by unit tests passing FakeGroq)
    if groq_service is not None:
        try:
            answer = groq_service.generate(
                system_prompt=GROUNDING_SYSTEM_PROMPT,
                user_prompt=_user_prompt(evaluation.query, context),
            )
            if isinstance(answer, str):
                answer = clean_generated_answer(answer)
            if (
                answer == NO_EVIDENCE_ANSWER
                and plan is not None
                and plan.retrieval_mode == "structured"
                and sources
            ):
                answer = structured_match_summary(sources)
            if not isinstance(answer, str) or not answer:
                raise GroqServiceError("Groq returned an empty response")
        except GroqServiceError:
            return RAGResult(
                query=evaluation.query,
                answer=None,
                decision="reject",
                reason="generation_unavailable",
                sources=sources,
                evaluation=evaluation,
            )
        return RAGResult(
            query=evaluation.query,
            answer=answer,
            decision="allow",
            reason="grounded_answer_generated",
            sources=sources,
            evaluation=evaluation,
        )

    # 2. LangChain Grounded Generation Chain (production path)
    try:
        model = chat_model or get_chat_model(temperature=0.0)
        is_structured = bool(plan is not None and plan.retrieval_mode == "structured")
        answer, reason = generate_grounded_answer(
            query=evaluation.query,
            chunks=evaluation.retrieved_chunks,
            chat_model=model,
            is_structured_search=is_structured,
            sources=sources,
        )

        if not answer or reason == "generation_unavailable":
            return RAGResult(
                query=evaluation.query,
                answer=None,
                decision="reject",
                reason="generation_unavailable",
                sources=sources,
                evaluation=evaluation,
            )

        if config.RAG_DEBUG:
            logger.info("LangChain generation succeeded with %d sources", len(sources))

        return RAGResult(
            query=evaluation.query,
            answer=answer,
            decision="allow",
            reason="grounded_answer_generated",
            sources=sources,
            evaluation=evaluation,
        )
    except Exception as exc:
        logger.error("LangChain grounded generation error: %s", exc)
        return RAGResult(
            query=evaluation.query,
            answer=None,
            decision="reject",
            reason="generation_unavailable",
            sources=sources,
            evaluation=evaluation,
        )
