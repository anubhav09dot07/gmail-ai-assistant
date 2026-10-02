from __future__ import annotations

from .schemas import GmailQuery, RetrievalPlan


class RetrievalRouter:
    """Deterministic router mapping structured query intents to retrieval plans."""

    def route(self, query: GmailQuery) -> RetrievalPlan:
        if query.intent == "out_of_scope":
            return RetrievalPlan.OUT_OF_SCOPE

        # Pure temporal / latest-email query without heavy semantic needs
        if query.intent == "latest_email":
            if not query.requires_semantic_search:
                return RetrievalPlan.GMAIL_AND_METADATA
            return RetrievalPlan.GMAIL_AND_VECTOR

        # Count or date lookup queries
        if query.intent in {"count", "date_lookup"}:
            if query.requires_gmail_search:
                return RetrievalPlan.GMAIL_AND_METADATA
            return RetrievalPlan.METADATA_ONLY

        # Sender lookup
        if query.intent == "sender_lookup":
            if query.requires_semantic_search:
                return RetrievalPlan.HYBRID
            return RetrievalPlan.GMAIL_AND_METADATA

        # List emails
        if query.intent == "list_emails":
            if query.requires_semantic_search:
                return RetrievalPlan.HYBRID
            return RetrievalPlan.GMAIL_AND_METADATA

        # Semantic questions, thread analysis, summaries
        if query.intent in {"semantic_question", "thread_question", "summary", "search"}:
            return RetrievalPlan.HYBRID

        if query.requires_gmail_search and not query.requires_semantic_search:
            return RetrievalPlan.GMAIL_ONLY

        if query.requires_semantic_search and not query.requires_gmail_search and not query.requires_metadata_search:
            return RetrievalPlan.VECTOR_ONLY

        return RetrievalPlan.HYBRID
