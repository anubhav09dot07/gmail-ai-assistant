from __future__ import annotations

import re
from typing import Sequence

from .. import config
from ..query.schemas import GmailQuery
from ..vector_store import RetrievedChunk
from .fusion import FusedCandidate

_TOKEN_PATTERN = re.compile(r"[a-z0-9']+")


class CandidateReranker:
    """Fast, local, zero-dependency candidate reranker evaluating multi-signal evidence match."""

    def __init__(self, enabled: bool | None = None) -> None:
        self.enabled = enabled if enabled is not None else config.RERANKER_ENABLED

    def rerank(
        self,
        candidates: Sequence[FusedCandidate | RetrievedChunk],
        query: GmailQuery,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """Rerank candidates based on query intent, exact sender/entity matches, and recency."""
        if not candidates:
            return []

        # Standardize items to RetrievedChunk with initial score
        items: list[tuple[RetrievedChunk, float]] = []
        for item in candidates:
            if isinstance(item, FusedCandidate):
                items.append((item.chunk, item.fused_score))
            else:
                items.append((item, item.score))

        if not self.enabled:
            # Sort by original score and return top_k
            items.sort(key=lambda x: x[1], reverse=True)
            return [chunk for chunk, _ in items[:top_k]]

        query_tokens = set(_TOKEN_PATTERN.findall(query.normalized_query.lower()))
        target_sender = query.sender.lower() if query.sender else ""
        target_entities = [e.lower() for e in query.entities]

        max_ts = max(
            (int(chunk.chunk.get("internal_date", 0) or 0) for chunk, _ in items),
            default=1,
        )
        min_ts = min(
            (int(chunk.chunk.get("internal_date", 0) or 0) for chunk, _ in items),
            default=0,
        )
        ts_span = max(1, max_ts - min_ts)

        reranked: list[tuple[RetrievedChunk, float]] = []
        for chunk, base_score in items:
            score_adj = base_score

            chunk_subject = str(chunk.chunk.get("subject", "")).lower()
            chunk_sender = str(chunk.chunk.get("sender", "")).lower()
            chunk_text = str(chunk.chunk.get("text", "")).lower()
            internal_date = int(chunk.chunk.get("internal_date", 0) or 0)

            # 1. Exact sender boost
            if target_sender and target_sender in chunk_sender:
                score_adj += 0.35

            # 2. Entity matches in subject or body
            for entity in target_entities:
                if entity in chunk_subject:
                    score_adj += 0.25
                elif entity in chunk_text:
                    score_adj += 0.15

            # 3. Subject term overlap
            subject_tokens = set(_TOKEN_PATTERN.findall(chunk_subject))
            if query_tokens and subject_tokens:
                overlap = len(query_tokens & subject_tokens) / len(query_tokens)
                score_adj += 0.20 * overlap

            # 4. Temporal priority if newest is desired
            if query.sort_order == "newest" and internal_date:
                recency_norm = (internal_date - min_ts) / ts_span
                score_adj += 0.25 * recency_norm

            reranked.append((chunk, score_adj))

        reranked.sort(key=lambda x: x[1], reverse=True)
        return [chunk for chunk, _ in reranked[:top_k]]
