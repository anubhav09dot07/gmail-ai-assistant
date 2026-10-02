from __future__ import annotations

from typing import Sequence

from .. import config
from ..query.schemas import GmailQuery
from ..vector_store import RetrievedChunk


def validate_evidence(
    query: GmailQuery,
    chunks: Sequence[RetrievedChunk],
    threshold: float | None = None,
    is_structured_search: bool = False,
) -> tuple[bool, float, str]:
    """Transparent multi-signal evidence validation.

    Considers:
    - Top semantic similarity
    - Exact sender matches
    - Temporal relevance
    - Structured search verification
    - Candidate volume
    """
    applied_threshold = threshold if threshold is not None else config.GMAIL_EVIDENCE_THRESHOLD

    if not chunks:
        return False, 0.0, "insufficient_gmail_evidence"

    top_chunk = chunks[0]
    base_score = float(top_chunk.score)

    evidence_score = base_score

    # Bonus for verified direct Gmail search hit
    if is_structured_search or query.requires_gmail_search:
        evidence_score = max(evidence_score, 0.85)

    # Bonus for exact sender match
    if query.sender:
        top_sender = str(top_chunk.chunk.get("sender", "")).lower()
        if query.sender.lower() in top_sender:
            evidence_score = max(evidence_score, min(1.0, evidence_score + 0.15))

    # Bonus for temporal match when sorting by newest
    if query.sort_order == "newest" and top_chunk.chunk.get("internal_date"):
        evidence_score = max(evidence_score, min(1.0, evidence_score + 0.05))

    is_sufficient = evidence_score >= applied_threshold
    reason = "evidence_sufficient" if is_sufficient else "insufficient_gmail_evidence"
    return is_sufficient, evidence_score, reason
