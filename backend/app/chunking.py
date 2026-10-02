import hashlib
import re

from . import config
from .models import NormalizedGmailMessage


class EmailChunk(dict):
    """A searchable email chunk with source metadata attached."""

    @property
    def text(self) -> str:
        return self["text"]


def _units(text: str) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    units: list[str] = []
    for paragraph in paragraphs:
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        units.extend(sentence.strip() for sentence in sentences if sentence.strip())
    return units


def _hard_split(text: str, size: int) -> list[str]:
    return [text[start : start + size].strip() for start in range(0, len(text), size)]


def _body_chunks(body: str, size: int, overlap: int) -> list[str]:
    units = _units(body)
    if not units:
        return []
    expanded: list[str] = []
    for unit in units:
        expanded.extend(_hard_split(unit, size) if len(unit) > size else [unit])

    chunks: list[str] = []
    current = ""
    for unit in expanded:
        candidate = f"{current}\n\n{unit}" if current else unit
        if current and len(candidate) > size:
            chunks.append(current)
            overlap_text = current[-overlap:].strip() if overlap else ""
            if overlap_text:
                available = max(1, size - len(overlap_text) - 2)
                current = f"{overlap_text}\n\n{unit[:available]}".strip()
                remainder = unit[available:].strip()
                if remainder:
                    chunks.extend(_body_chunks(remainder, size, overlap))
                    current = ""
            else:
                current = unit
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _chunk_id(message_id: str, index: int, text: str) -> str:
    digest = hashlib.sha256(f"{message_id}:{index}:{text}".encode()).hexdigest()[:16]
    return f"{message_id}:{index}:{digest}"


def chunk_message(
    message: NormalizedGmailMessage,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[EmailChunk]:
    size = chunk_size if chunk_size is not None else config.CHUNK_SIZE
    overlap_size = (
        overlap
        if overlap is not None
        else min(config.CHUNK_OVERLAP, max(0, size // 5))
    )
    if size <= 0 or overlap_size < 0 or overlap_size >= size:
        raise ValueError("chunk size must be positive and overlap must be smaller than size")
    body = message.body.strip()
    if not body:
        return []

    context = f"Subject: {message.subject}\nFrom: {message.sender}"
    if message.recipients:
        context += f"\nTo: {', '.join(message.recipients)}"
    content_size = max(1, size - len(context) - 2)
    body_chunks = _body_chunks(body, content_size, min(overlap_size, content_size - 1))
    total = len(body_chunks)
    return [
        EmailChunk(
            chunk_id=_chunk_id(message.message_id, index, body_text),
            message_id=message.message_id,
            thread_id=message.thread_id,
            subject=message.subject,
            sender=message.sender,
            recipients=list(message.recipients),
            timestamp=message.timestamp,
            date=message.date,
            labels=list(message.labels),
            snippet=message.snippet,
            chunk_index=index,
            total_chunks=total,
            text=f"{context}\n\n{body_text}",
            internal_date=getattr(message, "internal_date", 0),
        )
        for index, body_text in enumerate(body_chunks)
    ]


def chunk_messages(
    messages: list[NormalizedGmailMessage],
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[EmailChunk]:
    chunks: list[EmailChunk] = []
    for message in messages:
        chunks.extend(chunk_message(message, chunk_size=chunk_size, overlap=overlap))
    return chunks
