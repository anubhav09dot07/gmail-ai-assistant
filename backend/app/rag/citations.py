from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from ..vector_store import RetrievedChunk


@dataclass(frozen=True)
class RAGSource:
    subject: str
    sender: str
    date: str
    message_id: str
    thread_id: str
    score: float
    labels: tuple[str, ...] = ()
    gmail_url: str = ""
    internal_date: int = 0
    snippet: str = ""           # raw retrieval snippet
    citation_id: str = ""
    display_snippet: str = ""  # cleaned human-readable snippet for UI

    def public_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "sender": self.sender,
            "date": self.date,
            "message_id": self.message_id,
            "thread_id": self.thread_id,
            "score": self.score,
            "labels": list(self.labels),
            "gmail_url": self.gmail_url,
            "internal_date": self.internal_date,
            "snippet": self.display_snippet or self.snippet,
        }


def gmail_url(message_id: str) -> str:
    if not message_id:
        return ""
    return f"https://mail.google.com/mail/u/0/#inbox/{message_id}"


def source_from_chunk(result: RetrievedChunk, index: int = 1) -> RAGSource:
    chunk = result.chunk
    msg_id = str(chunk.get("message_id", ""))
    raw_snippet = str(chunk.get("snippet", ""))
    # Attempt to use a pre-cleaned display snippet if the chunk carries one.
    display_snippet = str(chunk.get("display_snippet", "")) or raw_snippet
    return RAGSource(
        subject=str(chunk.get("subject", "")),
        sender=str(chunk.get("sender", "")),
        date=str(chunk.get("date", "")),
        message_id=msg_id,
        thread_id=str(chunk.get("thread_id", "")),
        score=result.score,
        labels=tuple(chunk.get("labels", []) or ()),
        gmail_url=gmail_url(msg_id),
        internal_date=int(chunk.get("internal_date", 0) or 0),
        snippet=raw_snippet,
        citation_id=f"E{index}",
        display_snippet=display_snippet,
    )


def deduplicate_sources(results: Sequence[RetrievedChunk]) -> tuple[RAGSource, ...]:
    """Expose one strongest source per message without merging same-thread messages."""
    selected: dict[tuple[str, str], RAGSource] = {}
    for index, result in enumerate(results, start=1):
        source = source_from_chunk(result, index=index)
        if source.message_id:
            key = ("message", source.message_id)
        elif source.thread_id:
            key = ("thread", source.thread_id)
        else:
            key = ("chunk", str(result.chunk.get("chunk_id", "")))
        current = selected.get(key)
        if current is None or source.score > current.score:
            selected[key] = source
    return tuple(selected.values())
