from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from . import config
from .chunking import chunk_message
from .embeddings import EmbeddingService
from .gmail import fetch_messages_paginated
from .models import NormalizedGmailMessage
from .vector_store import QdrantVectorStore


@dataclass(frozen=True)
class SyncStats:
    messages_discovered: int
    messages_new: int
    messages_updated: int
    messages_unchanged: int
    messages_deleted: int
    chunks_created: int
    chunks_removed: int
    complete_listing: bool
    duration_seconds: float = 0.0

    def public_dict(self) -> dict[str, Any]:
        return asdict(self)


class SyncMetadata:
    """Small atomic JSON index of message fingerprints, scoped by user ID."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or config.SYNC_METADATA_FILE

    def load(self) -> dict[str, dict[str, dict[str, Any]]]:
        if not self.path.is_file():
            return {}
        data = json.loads(self.path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}

    def save(self, data: dict[str, dict[str, dict[str, Any]]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(f"{self.path.suffix}.tmp")
        temporary.write_text(json.dumps(data, sort_keys=True, indent=2), encoding="utf-8")
        temporary.replace(self.path)


def _fingerprint(message: NormalizedGmailMessage) -> str:
    payload = {
        "message_id": message.message_id,
        "thread_id": message.thread_id,
        "subject": message.subject,
        "sender": message.sender,
        "recipients": message.recipients,
        "timestamp": message.timestamp,
        "date": message.date,
        "body": message.body,
        "snippet": message.snippet,
        "labels": message.labels,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class GmailSyncService:
    def __init__(
        self,
        *,
        vector_store: QdrantVectorStore | None = None,
        embedding_service: EmbeddingService | None = None,
        metadata: SyncMetadata | None = None,
        fetcher: Callable[..., tuple[list[NormalizedGmailMessage], bool]] = fetch_messages_paginated,
    ) -> None:
        self.vector_store = vector_store or QdrantVectorStore()
        self.embedding_service = embedding_service or EmbeddingService()
        self.metadata = metadata or SyncMetadata()
        self.fetcher = fetcher

    def sync(
        self,
        user_id: str,
        *,
        max_messages: int | None = None,
        page_size: int | None = None,
        service=None,
    ) -> SyncStats:
        if not user_id or not user_id.strip():
            raise ValueError("user_id is required for sync")
        started = time.perf_counter()
        messages, complete_listing = self.fetcher(
            max_messages=max_messages or config.GMAIL_SYNC_MAX_MESSAGES,
            page_size=page_size or config.GMAIL_SYNC_PAGE_SIZE,
            service=service,
        )
        state = self.metadata.load()
        user_state = state.setdefault(user_id, {})
        current_ids = {message.message_id for message in messages if message.message_id}
        messages_new = messages_updated = messages_unchanged = 0
        chunks_created = chunks_removed = 0

        for message in messages:
            if not message.message_id:
                continue
            fingerprint = _fingerprint(message)
            previous = user_state.get(message.message_id)
            if previous and previous.get("fingerprint") == fingerprint:
                messages_unchanged += 1
                continue
            if previous:
                messages_updated += 1
            else:
                messages_new += 1
            chunks = chunk_message(message)
            embedded_chunks = self.embedding_service.embed_chunks(chunks)
            if previous:
                self.vector_store.delete_message(message.message_id, user_id=user_id)
                chunks_removed += int(previous.get("chunk_count", 0))
            if chunks:
                chunks_created += self.vector_store.upsert(embedded_chunks, user_id=user_id)
            user_state[message.message_id] = {
                "fingerprint": fingerprint,
                "thread_id": message.thread_id,
                "chunk_count": len(chunks),
            }

        messages_deleted = 0
        if complete_listing:
            for message_id in set(user_state) - current_ids:
                previous = user_state.pop(message_id)
                self.vector_store.delete_message(message_id, user_id=user_id)
                chunks_removed += int(previous.get("chunk_count", 0))
                messages_deleted += 1

        self.metadata.save(state)
        elapsed = time.perf_counter() - started
        return SyncStats(
            messages_discovered=len(messages),
            messages_new=messages_new,
            messages_updated=messages_updated,
            messages_unchanged=messages_unchanged,
            messages_deleted=messages_deleted,
            chunks_created=chunks_created,
            chunks_removed=chunks_removed,
            complete_listing=complete_listing,
            duration_seconds=elapsed,
        )
