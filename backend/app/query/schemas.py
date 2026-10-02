from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal
from pydantic import BaseModel, Field


class RetrievalPlan(str, Enum):
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    GMAIL_ONLY = "GMAIL_ONLY"
    VECTOR_ONLY = "VECTOR_ONLY"
    METADATA_ONLY = "METADATA_ONLY"
    GMAIL_AND_METADATA = "GMAIL_AND_METADATA"
    GMAIL_AND_VECTOR = "GMAIL_AND_VECTOR"
    HYBRID = "HYBRID"


class GmailQuery(BaseModel):
    """Strongly typed representation of a user query in the Gmail assistant."""

    original_query: str
    normalized_query: str = ""
    intent: Literal[
        "semantic_question",
        "latest_email",
        "search",
        "list_emails",
        "count",
        "summary",
        "thread_question",
        "sender_lookup",
        "date_lookup",
        "out_of_scope",
    ] = "semantic_question"

    entities: list[str] = Field(default_factory=list)
    sender: str | None = None
    recipients: list[str] = Field(default_factory=list)
    subject: str | None = None
    label: str | None = None
    keywords: list[str] = Field(default_factory=list)

    time_start: datetime | None = None
    time_end: datetime | None = None
    sort_order: Literal["newest", "oldest", "relevance"] = "relevance"

    requires_semantic_search: bool = True
    requires_gmail_search: bool = False
    requires_metadata_search: bool = False

    extracted_topic: str = ""
    expanded_terms: list[str] = Field(default_factory=list)
    gmail_query_string: str = ""
    is_contextual_question: bool = False
