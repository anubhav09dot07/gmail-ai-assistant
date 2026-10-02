from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models

from . import config
from .chunking import EmailChunk, chunk_messages
from .embeddings import EmbeddedChunk, EmbeddingService
from .models import NormalizedGmailMessage


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: EmailChunk
    score: float


class QdrantVectorStore:
    """Persistent Qdrant storage for metadata-preserving email chunks."""

    def __init__(
        self,
        client: QdrantClient | None = None,
        url: str | None = None,
        collection_name: str | None = None,
        vector_size: int | None = None,
        batch_size: int | None = None,
    ) -> None:
        self.client = client or QdrantClient(url=url or config.QDRANT_URL)
        self.collection_name = collection_name or config.QDRANT_COLLECTION
        self.vector_size = vector_size or config.EMBEDDING_DIMENSION
        self.batch_size = batch_size or config.QDRANT_BATCH_SIZE
        if self.vector_size <= 0 or self.batch_size <= 0:
            raise ValueError("vector size and batch size must be positive")

    def ensure_collection(self) -> None:
        """Create the collection once, or reject an incompatible existing one."""
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size, distance=models.Distance.COSINE
                ),
            )
            return

        info = self.client.get_collection(self.collection_name)
        vectors = info.config.params.vectors
        size = getattr(vectors, "size", None)
        distance = getattr(vectors, "distance", None)
        if size != self.vector_size or distance != models.Distance.COSINE:
            raise ValueError(
                f"Qdrant collection {self.collection_name!r} must use "
                f"{self.vector_size} dimensions and COSINE distance"
            )

    @staticmethod
    def _point_id(user_id: str, chunk_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"gmail-chunk:{user_id}:{chunk_id}"))

    @staticmethod
    def _payload(chunk: EmailChunk, user_id: str) -> dict[str, Any]:
        return {
            "user_id": user_id,
            "chunk_id": chunk["chunk_id"],
            "message_id": chunk["message_id"],
            "thread_id": chunk["thread_id"],
            "subject": chunk["subject"],
            "sender": chunk["sender"],
            "recipients": chunk["recipients"],
            "timestamp": chunk["timestamp"],
            "date": chunk["date"],
            "snippet": chunk.get("snippet", ""),
            "labels": chunk["labels"],
            "chunk_index": chunk["chunk_index"],
            "total_chunks": chunk["total_chunks"],
            "text": chunk.text,
            "internal_date": chunk.get("internal_date", 0),
        }

    def upsert(self, embedded_chunks: list[EmbeddedChunk], *, user_id: str) -> int:
        """Batch-upsert chunks; repeating the same chunks is idempotent."""
        if not user_id:
            raise ValueError("user_id is required for indexing")
        self.ensure_collection()
        for embedded in embedded_chunks:
            if len(embedded.vector) != self.vector_size:
                raise ValueError(
                    f"expected vectors of size {self.vector_size}, "
                    f"got {len(embedded.vector)}"
                )

        for start in range(0, len(embedded_chunks), self.batch_size):
            batch = embedded_chunks[start : start + self.batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=self._point_id(user_id, item.chunk["chunk_id"]),
                        vector=item.vector,
                        payload=self._payload(item.chunk, user_id),
                    )
                    for item in batch
                ],
                wait=True,
            )
        return len(embedded_chunks)

    def index_chunks(
        self,
        chunks: list[EmailChunk],
        embedding_service: EmbeddingService,
        *,
        user_id: str,
    ) -> int:
        """Embed a chunk batch once and persist all resulting vectors."""
        return self.upsert(embedding_service.embed_chunks(chunks), user_id=user_id)

    def index_messages(
        self,
        messages: list[NormalizedGmailMessage],
        embedding_service: EmbeddingService,
        *,
        user_id: str,
    ) -> int:
        """Chunk, batch-embed, and persist normalized Gmail messages."""
        return self.index_chunks(
            chunk_messages(messages), embedding_service, user_id=user_id
        )

    @staticmethod
    def _match(field: str, value: str) -> models.FieldCondition:
        return models.FieldCondition(key=field, match=models.MatchValue(value=value))

    def search(
        self,
        query_vector: list[float],
        *,
        user_id: str,
        limit: int = 10,
        sender: str | None = None,
        subject: str | None = None,
        thread_id: str | None = None,
        label: str | None = None,
        labels: list[str] | tuple[str, ...] | None = None,
        internal_date_after: int | None = None,
        internal_date_before: int | None = None,
    ) -> list[RetrievedChunk]:
        """Return similarity-ranked, user-scoped chunks with optional filters."""
        if not user_id:
            raise ValueError("user_id is required for retrieval")
        if len(query_vector) != self.vector_size:
            raise ValueError(f"expected a query vector of size {self.vector_size}")
        if limit <= 0:
            raise ValueError("limit must be positive")

        conditions: list[models.FieldCondition] = [self._match("user_id", user_id)]
        for field, value in (
            ("sender", sender),
            ("subject", subject),
            ("thread_id", thread_id),
        ):
            if value:
                conditions.append(self._match(field, value))
        if label:
            conditions.append(
                models.FieldCondition(
                    key="labels", match=models.MatchValue(value=label)
                )
            )
        if labels:
            for l in labels:
                conditions.append(
                    models.FieldCondition(
                        key="labels", match=models.MatchValue(value=l)
                    )
                )
        if internal_date_after is not None or internal_date_before is not None:
            conditions.append(
                models.FieldCondition(
                    key="internal_date",
                    range=models.Range(
                        gte=internal_date_after, lte=internal_date_before
                    ),
                )
            )

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=models.Filter(must=conditions),
            limit=limit,
            with_payload=True,
        )
        return [
            RetrievedChunk(
                chunk=EmailChunk(point.payload or {}), score=float(point.score)
            )
            for point in response.points
        ]

    def scroll_chunks(
        self,
        *,
        user_id: str,
        limit: int | None = None,
        page_size: int = 256,
        sender: str | None = None,
        labels: list[str] | tuple[str, ...] | None = None,
        internal_date_after: int | None = None,
        internal_date_before: int | None = None,
    ) -> list[RetrievedChunk]:
        """Return user-scoped payloads over all pages, or an explicit caller limit.

        This is intentionally a diagnostic/fallback API. Query-time retrieval should
        prefer Gmail's native search or Qdrant similarity search; callers must never
        accidentally receive only the first arbitrary 1,000 chunks.
        """
        if not user_id:
            raise ValueError("user_id is required for retrieval")
        if limit is not None and limit <= 0:
            raise ValueError("limit must be positive when provided")
        if page_size <= 0:
            raise ValueError("page_size must be positive")

        must_conditions: list[models.FieldCondition] = [self._match("user_id", user_id)]
        if sender:
            must_conditions.append(self._match("sender", sender))
        if labels:
            for l in labels:
                must_conditions.append(
                    models.FieldCondition(
                        key="labels", match=models.MatchValue(value=l)
                    )
                )
        if internal_date_after is not None or internal_date_before is not None:
            must_conditions.append(
                models.FieldCondition(
                    key="internal_date",
                    range=models.Range(
                        gte=internal_date_after, lte=internal_date_before
                    ),
                )
            )

        records: list[RetrievedChunk] = []
        offset = None
        while limit is None or len(records) < limit:
            request_size = page_size if limit is None else min(page_size, limit - len(records))
            page, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(must=must_conditions),
                limit=request_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            records.extend(
                RetrievedChunk(chunk=EmailChunk(record.payload or {}), score=0.0)
                for record in page
            )
            if next_offset is None or not page:
                break
            offset = next_offset
        return records

    def delete_message(self, message_id: str, *, user_id: str) -> None:
        """Remove all chunks for one user-owned Gmail message."""
        if not user_id:
            raise ValueError("user_id is required for deletion")
        self.ensure_collection()
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        self._match("user_id", user_id),
                        self._match("message_id", message_id),
                    ]
                )
            ),
            wait=True,
        )

    def count(self, *, user_id: str | None = None) -> int:
        self.ensure_collection()
        query_filter = (
            models.Filter(must=[self._match("user_id", user_id)])
            if user_id
            else None
        )
        return int(
            self.client.count(
                collection_name=self.collection_name,
                count_filter=query_filter,
                exact=True,
            ).count
        )
