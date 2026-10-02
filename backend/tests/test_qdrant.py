import unittest

from qdrant_client import QdrantClient

from backend.app.chunking import EmailChunk
from backend.app.embeddings import EmbeddedChunk
from backend.app.vector_store import QdrantVectorStore


def chunk(chunk_id: str, subject: str = "Fee deadline") -> EmailChunk:
    return EmailChunk(
        chunk_id=chunk_id,
        message_id=f"message-{chunk_id}",
        thread_id="thread-1",
        subject=subject,
        sender="university@example.com",
        recipients=["student@example.com"],
        timestamp="2026-09-05T12:00:00+00:00",
        date="2026-09-05T12:00:00+00:00",
        labels=["INBOX", "IMPORTANT"],
        chunk_index=0,
        total_chunks=1,
        text=f"Subject: {subject}\n\nThe deadline is 10 September.",
    )


class QdrantVectorStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = QdrantVectorStore(
            client=QdrantClient(location=":memory:"),
            collection_name="gmail_knowledge",
            vector_size=3,
            batch_size=1,
        )

    def test_batch_upsert_is_deterministic_and_metadata_filterable(self) -> None:
        items = [
            EmbeddedChunk(chunk("one"), [1.0, 0.0, 0.0]),
            EmbeddedChunk(chunk("two", subject="Travel plans"), [0.0, 1.0, 0.0]),
        ]
        self.assertEqual(self.store.upsert(items, user_id="user-1"), 2)
        self.assertEqual(self.store.upsert(items, user_id="user-1"), 2)
        self.assertEqual(self.store.count(user_id="user-1"), 2)

        results = self.store.search(
            [1.0, 0.0, 0.0], user_id="user-1", subject="Fee deadline"
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk["chunk_id"], "one")
        self.assertEqual(results[0].chunk["user_id"], "user-1")

    def test_user_scope_is_required_and_isolated(self) -> None:
        item = EmbeddedChunk(chunk("one"), [1.0, 0.0, 0.0])
        self.store.upsert([item], user_id="user-1")
        self.store.upsert([item], user_id="user-2")
        self.assertEqual(self.store.count(user_id="user-1"), 1)
        self.assertEqual(self.store.search([1.0, 0.0, 0.0], user_id="user-2")[0].chunk["user_id"], "user-2")
        with self.assertRaises(ValueError):
            self.store.search([1.0, 0.0, 0.0], user_id="")


if __name__ == "__main__":
    unittest.main()