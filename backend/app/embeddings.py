import math
from dataclasses import dataclass
from typing import Any

from . import config
from .chunking import EmailChunk


@dataclass(frozen=True)
class EmbeddedChunk:
    chunk: EmailChunk
    vector: list[float]


class EmbeddingService:
    """Lazy local Sentence Transformers service for batch embeddings."""

    def __init__(self, model_name: str | None = None, model: Any = None) -> None:
        self.model_name = model_name or config.EMBEDDING_MODEL
        self._model = model

    @property
    def model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dimension(self) -> int:
        dimension = self.model.get_sentence_embedding_dimension()
        if dimension is None:
            raise RuntimeError("Embedding model did not report a vector dimension")
        return int(dimension)

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        encoded = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        vectors = [[float(value) for value in row] for row in encoded]
        if any(not math.isfinite(value) for vector in vectors for value in vector):
            raise ValueError("Embedding model returned a non-finite value")
        return vectors

    def embed_chunks(self, chunks: list[EmailChunk]) -> list[EmbeddedChunk]:
        vectors = self.embed_texts([chunk.text for chunk in chunks])
        return [
            EmbeddedChunk(chunk=chunk, vector=vector)
            for chunk, vector in zip(chunks, vectors)
        ]