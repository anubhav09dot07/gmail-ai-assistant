from __future__ import annotations

from typing import Any
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field

from ..embeddings import EmbeddingService
from ..vector_store import QdrantVectorStore, RetrievedChunk


class GmailQdrantRetriever(BaseRetriever):
    """LangChain-compatible retriever wrapping QdrantVectorStore with mandatory user isolation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    user_id: str = Field(description="Mandatory user ID for strict multi-tenant isolation")
    vector_store: Any = Field(default=None, description="QdrantVectorStore instance")
    embedding_service: Any = Field(default=None, description="EmbeddingService instance")
    limit: int = Field(default=5, description="Maximum chunks to retrieve")
    sender: str | None = Field(default=None, description="Optional sender filter")
    subject: str | None = Field(default=None, description="Optional subject filter")
    label: str | None = Field(default=None, description="Optional Gmail label filter")
    thread_id: str | None = Field(default=None, description="Optional thread ID filter")
    date_from: str | None = Field(default=None, description="Optional start date ISO string")
    date_to: str | None = Field(default=None, description="Optional end date ISO string")

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not self.user_id or not self.user_id.strip():
            raise ValueError("user_id is mandatory and cannot be empty for GmailQdrantRetriever")
        if self.vector_store is None:
            self.vector_store = QdrantVectorStore()
        if self.embedding_service is None:
            self.embedding_service = EmbeddingService()

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> list[Document]:
        """Execute dense semantic search in Qdrant scoped to user_id."""
        query_vector = self.embedding_service.embed_query(query)
        chunks: list[RetrievedChunk] = self.vector_store.search(
            query_vector,
            user_id=self.user_id,
            sender=self.sender,
            subject=self.subject,
            label=self.label,
            date_from=self.date_from,
            date_to=self.date_to,
            thread_id=self.thread_id,
            limit=self.limit,
        )
        documents: list[Document] = []
        for item in chunks:
            payload = dict(item.chunk)
            content = payload.get("text", "")
            metadata = {k: v for k, v in payload.items() if k != "text"}
            metadata["score"] = item.score
            documents.append(Document(page_content=content, metadata=metadata))
        return documents

    @staticmethod
    def chunk_to_document(chunk: RetrievedChunk) -> Document:
        payload = dict(chunk.chunk)
        content = payload.get("text", "")
        metadata = {k: v for k, v in payload.items() if k != "text"}
        metadata["score"] = chunk.score
        return Document(page_content=content, metadata=metadata)
