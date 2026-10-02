import unittest
import uuid

from qdrant_client import QdrantClient

from backend.app.chunking import EmailChunk
from backend.app.embeddings import EmbeddedChunk
from backend.app.vector_store import QdrantVectorStore


class QdrantIntegrationTests(unittest.TestCase):
    def test_existing_qdrant_round_trip(self) -> None:
        client = QdrantClient(url="http://localhost:6333")
        collection = "gmail_knowledge"
        store = QdrantVectorStore(client=client, collection_name=collection, batch_size=2)
        store.ensure_collection()
        user_id = f"phase-4-test-{uuid.uuid4()}"
        chunk = EmailChunk(
            chunk_id=f"chunk-{uuid.uuid4()}", message_id="integration-message",
            thread_id="integration-thread", subject="Integration deadline",
            sender="integration@example.com", recipients=["user@example.com"],
            timestamp="2026-09-05T12:00:00+00:00", date="2026-09-05T12:00:00+00:00",
            labels=["INBOX"], chunk_index=0, total_chunks=1,
            text="Subject: Integration deadline\n\nSubmit by Friday.",
        )
        try:
            self.assertEqual(
                store.upsert(
                    [EmbeddedChunk(chunk, [1.0] + [0.0] * 383)], user_id=user_id
                ),
                1,
            )
            results = store.search([1.0] + [0.0] * 383, user_id=user_id, limit=1)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].chunk["subject"], "Integration deadline")
            self.assertGreaterEqual(results[0].score, 0.99)
        finally:
            store.delete_message("integration-message", user_id=user_id)


if __name__ == "__main__":
    unittest.main()