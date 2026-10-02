from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from .. import config
from ..llm.factory import get_chat_model
from ..query_normalizer import normalize_query
from ..query_planner import SearchPlan, _check_relevance, plan_search
from .schemas import GmailQuery

logger = logging.getLogger(__name__)

QUERY_UNDERSTANDING_SYSTEM_PROMPT = """You are a query analysis assistant for a personal Gmail Knowledge Assistant.
Your task is to analyze the user's natural language question and extract a structured query representation.

RULES:
1. The assistant operates ONLY on Gmail data.
2. If the user question is completely unrelated to personal email (e.g. general trivia, coding instructions, weather, general world knowledge), classify intent as "out_of_scope".
3. Extract explicit entities, topics, sender constraints, and temporal constraints conservatively.
4. NEVER invent or assume email addresses or dates that were not requested or implied.
5. If the user asks for the latest/last/most recent email, set intent to "latest_email" and sort_order to "newest".
6. If the user asks to list or show emails, set intent to "list_emails".
7. If the user asks to count emails, set intent to "count".
8. If the user asks about specific factual content/decisions/amounts inside emails, set intent to "semantic_question".
9. Set requires_semantic_search=True whenever semantic meaning of the body text is needed to answer.
10. Set requires_gmail_search=True if searching specific senders, dates, labels, or exact terms in Gmail API is helpful.
11. Set requires_metadata_search=True if metadata filtering (dates, senders, read/unread) is sufficient.
"""

class LLMQueryExtraction(BaseModel):
    intent: str = Field(
        default="semantic_question",
        description="One of: semantic_question, latest_email, search, list_emails, count, summary, thread_question, sender_lookup, date_lookup, out_of_scope"
    )
    entities: list[str] = Field(default_factory=list, description="Named entities mentioned in the query")
    sender: str | None = Field(default=None, description="Sender name or domain if query restricts by sender")
    recipients: list[str] = Field(default_factory=list, description="Recipient names if specified")
    keywords: list[str] = Field(default_factory=list, description="Key search terms or topics")
    sort_order: str = Field(default="relevance", description="One of: newest, oldest, relevance")
    requires_semantic_search: bool = Field(default=True, description="Whether vector search over email bodies is needed")
    requires_gmail_search: bool = Field(default=False, description="Whether direct Gmail API search is needed")
    requires_metadata_search: bool = Field(default=False, description="Whether metadata search is needed")


def _deterministic_intent(plan: SearchPlan) -> str:
    """Infer intent deterministically from search plan facets."""
    if not plan.is_gmail_related:
        return "out_of_scope"
    if plan.sort == "newest":
        return "latest_email"
    if plan.sender and not plan.semantic_query:
        return "sender_lookup"
    if plan.use_structured_search and not plan.use_semantic_search:
        return "search"
    return "semantic_question"


def _plan_to_gmail_query(plan: SearchPlan, raw_query: str, normalized: str) -> GmailQuery:
    intent = _deterministic_intent(plan)
    time_start = None
    time_end = None
    if plan.internal_date_after:
        time_start = datetime.fromtimestamp(plan.internal_date_after / 1000, tz=timezone.utc)
    if plan.internal_date_before:
        time_end = datetime.fromtimestamp(plan.internal_date_before / 1000, tz=timezone.utc)

    sort_order = "newest" if plan.sort == "newest" else "relevance"
    requires_semantic = plan.use_semantic_search or intent == "semantic_question"
    requires_gmail = plan.use_structured_search or bool(plan.gmail_query)
    requires_metadata = bool(plan.sender or plan.labels or plan.internal_date_after or plan.internal_date_before)

    return GmailQuery(
        original_query=raw_query,
        normalized_query=normalized,
        intent=intent,  # type: ignore[arg-type]
        entities=list(plan.entities),
        sender=plan.sender,
        subject=plan.subject,
        label=plan.labels[0] if plan.labels else None,
        keywords=[token for token in plan.semantic_query.split() if token] if plan.semantic_query else [],
        time_start=time_start,
        time_end=time_end,
        sort_order=sort_order,  # type: ignore[arg-type]
        requires_semantic_search=requires_semantic,
        requires_gmail_search=requires_gmail,
        requires_metadata_search=requires_metadata,
        extracted_topic=plan.semantic_query,
        gmail_query_string=plan.gmail_query,
    )


def understand_query(
    query: str,
    chat_model: BaseChatModel | None = None,
    allow_llm: bool = True,
) -> GmailQuery:
    """Analyze query using fast deterministic rules and optional LangChain LLM structured understanding."""
    clean_query = query.strip()
    normalized = normalize_query(clean_query)

    # 1. Cheap scope / intent detection (Gate 1 fast-fail)
    is_related, _ = _check_relevance(clean_query)
    if not is_related:
        if config.RAG_DEBUG:
            logger.info("Query %r rejected as out_of_scope by deterministic gate", clean_query)
        return GmailQuery(
            original_query=clean_query,
            normalized_query=normalized,
            intent="out_of_scope",
            requires_semantic_search=False,
            requires_gmail_search=False,
            requires_metadata_search=False,
        )

    # 2. Compute deterministic search plan
    plan = plan_search(clean_query)

    # 3. Check for obvious fast-path:
    # Queries like "latest email from X", pure sender searches, or direct date lookups
    # don't need a slow LLM call for query understanding.
    is_fast_path = (
        plan.sort == "newest"
        or (plan.sender is not None and not plan.semantic_query)
        or (plan.use_structured_search and not plan.use_semantic_search)
        or not allow_llm
        or not config.GROQ_API_KEY
    )

    if is_fast_path:
        base_query = _plan_to_gmail_query(plan, clean_query, normalized)
        if config.RAG_DEBUG:
            logger.info("Query understanding used deterministic fast-path: intent=%s", base_query.intent)
        return base_query

    # 4. Complex semantic query understanding using LangChain ChatModel + Structured Output
    try:
        model = chat_model or get_chat_model(temperature=0.0)
        structured_llm = model.with_structured_output(LLMQueryExtraction)
        prompt = ChatPromptTemplate.from_messages([
            ("system", QUERY_UNDERSTANDING_SYSTEM_PROMPT),
            ("user", "Analyze this user query: {query}"),
        ])
        chain = prompt | structured_llm
        extraction: LLMQueryExtraction = chain.invoke({"query": clean_query})

        # Harmonize LLM extraction with deterministic safety constraints
        valid_intents = {
            "semantic_question", "latest_email", "search", "list_emails",
            "count", "summary", "thread_question", "sender_lookup",
            "date_lookup", "out_of_scope"
        }
        chosen_intent = extraction.intent if extraction.intent in valid_intents else "semantic_question"
        if plan.sort == "newest":
            chosen_intent = "latest_email"

        sender = extraction.sender or plan.sender
        entities = list(dict.fromkeys(list(plan.entities) + extraction.entities))
        sort_order = "newest" if (extraction.sort_order == "newest" or plan.sort == "newest") else "relevance"

        time_start = None
        time_end = None
        if plan.internal_date_after:
            time_start = datetime.fromtimestamp(plan.internal_date_after / 1000, tz=timezone.utc)
        if plan.internal_date_before:
            time_end = datetime.fromtimestamp(plan.internal_date_before / 1000, tz=timezone.utc)

        return GmailQuery(
            original_query=clean_query,
            normalized_query=normalized,
            intent=chosen_intent,  # type: ignore[arg-type]
            entities=entities,
            sender=sender,
            subject=plan.subject,
            label=plan.labels[0] if plan.labels else None,
            keywords=extraction.keywords or ([t for t in plan.semantic_query.split()] if plan.semantic_query else []),
            time_start=time_start,
            time_end=time_end,
            sort_order=sort_order,  # type: ignore[arg-type]
            requires_semantic_search=extraction.requires_semantic_search or plan.use_semantic_search,
            requires_gmail_search=extraction.requires_gmail_search or plan.use_structured_search or bool(plan.gmail_query),
            requires_metadata_search=extraction.requires_metadata_search or bool(sender or plan.labels),
            extracted_topic=plan.semantic_query,
            gmail_query_string=plan.gmail_query,
        )
    except Exception as exc:
        logger.warning("LLM query understanding failed or unavailable (%s), falling back to deterministic plan", exc)
        return _plan_to_gmail_query(plan, clean_query, normalized)
