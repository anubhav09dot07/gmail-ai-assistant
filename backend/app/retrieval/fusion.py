from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .. import config
from ..vector_store import RetrievedChunk


@dataclass
class FusedCandidate:
    chunk: RetrievedChunk
    fused_score: float
    vector_rank: int | None = None
    lexical_rank: int | None = None
    gmail_rank: int | None = None
    temporal_rank: int | None = None


class ReciprocalRankFusion:
    """Configurable Reciprocal Rank Fusion (RRF) across multi-channel retrievers."""

    def __init__(
        self,
        k: int = 60,
        vector_weight: float | None = None,
        lexical_weight: float | None = None,
        gmail_weight: float | None = None,
        temporal_weight: float | None = None,
    ) -> None:
        self.k = k
        self.vector_weight = vector_weight if vector_weight is not None else config.RRF_WEIGHT_VECTOR
        self.lexical_weight = lexical_weight if lexical_weight is not None else config.RRF_WEIGHT_LEXICAL
        self.gmail_weight = gmail_weight if gmail_weight is not None else config.RRF_WEIGHT_GMAIL
        self.temporal_weight = temporal_weight if temporal_weight is not None else config.RRF_WEIGHT_TEMPORAL

    def fuse(
        self,
        *,
        vector_candidates: Sequence[RetrievedChunk] = (),
        lexical_candidates: Sequence[RetrievedChunk] = (),
        gmail_candidates: Sequence[RetrievedChunk] = (),
        sort_order: str = "relevance",
    ) -> list[FusedCandidate]:
        """Merge candidate lists from vector, lexical, and Gmail API channels."""
        # Key candidates by chunk_id
        candidate_map: dict[str, RetrievedChunk] = {}
        vector_ranks: dict[str, int] = {}
        lexical_ranks: dict[str, int] = {}
        gmail_ranks: dict[str, int] = {}

        for rank, item in enumerate(vector_candidates, start=1):
            cid = str(item.chunk.get("chunk_id", ""))
            candidate_map[cid] = item
            vector_ranks[cid] = rank

        for rank, item in enumerate(lexical_candidates, start=1):
            cid = str(item.chunk.get("chunk_id", ""))
            if cid not in candidate_map:
                candidate_map[cid] = item
            lexical_ranks[cid] = rank

        for rank, item in enumerate(gmail_candidates, start=1):
            cid = str(item.chunk.get("chunk_id", ""))
            if cid not in candidate_map:
                candidate_map[cid] = item
            gmail_ranks[cid] = rank

        # Calculate temporal rank if sort_order is newest
        temporal_ranks: dict[str, int] = {}
        if sort_order == "newest" and candidate_map:
            sorted_by_date = sorted(
                candidate_map.values(),
                key=lambda c: int(c.chunk.get("internal_date", 0) or 0),
                reverse=True,
            )
            for rank, item in enumerate(sorted_by_date, start=1):
                temporal_ranks[str(item.chunk.get("chunk_id", ""))] = rank

        fused: list[FusedCandidate] = []
        for cid, item in candidate_map.items():
            score = 0.0
            vr = vector_ranks.get(cid)
            lr = lexical_ranks.get(cid)
            gr = gmail_ranks.get(cid)
            tr = temporal_ranks.get(cid)

            if vr is not None:
                score += self.vector_weight * (1.0 / (self.k + vr))
            if lr is not None:
                score += self.lexical_weight * (1.0 / (self.k + lr))
            if gr is not None:
                score += self.gmail_weight * (1.0 / (self.k + gr))
            if tr is not None:
                score += self.temporal_weight * (1.0 / (self.k + tr))

            fused.append(
                FusedCandidate(
                    chunk=item,
                    fused_score=score,
                    vector_rank=vr,
                    lexical_rank=lr,
                    gmail_rank=gr,
                    temporal_rank=tr,
                )
            )

        fused.sort(key=lambda x: x.fused_score, reverse=True)
        return fused
