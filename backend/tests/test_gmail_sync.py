import tempfile
import unittest
from pathlib import Path

from backend.app.gmail import fetch_messages_paginated
from backend.app.gmail_sync_service import GmailSyncService, SyncMetadata
from backend.app.models import NormalizedGmailMessage
from backend.app.vector_store import QdrantVectorStore
from qdrant_client import QdrantClient


class FakeEmbeddingService:
    def __init__(self) -> None:
        self.calls = 0

    def embed_chunks(self, chunks):
        self.calls += len(chunks)
        from backend.app.embeddings import EmbeddedChunk
        return [EmbeddedChunk(chunk, [1.0, 0.0, 0.0]) for chunk in chunks]


def message(message_id: str, body: str, subject: str = "Subject") -> NormalizedGmailMessage:
    return NormalizedGmailMessage(
        message_id=message_id,
        thread_id=f"thread-{message_id}",
        subject=subject,
        sender="sender@example.com",
        recipients=["user@example.com"],
        timestamp="2026-09-05T12:00:00+00:00",
        date="2026-09-05T12:00:00+00:00",
        body=body,
        snippet=body[:30],
        labels=["INBOX"],
    )


class GmailSyncTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.store = QdrantVectorStore(
            client=QdrantClient(location=":memory:"),
            collection_name="sync-tests",
            vector_size=3,
            batch_size=10,
        )
        self.embedding = FakeEmbeddingService()
        self.state = SyncMetadata(Path(self.tempdir.name) / "sync.json")
        self.messages = [message("one", "First body."), message("two", "Second body.")]

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def fetcher(self, **kwargs):
        return list(self.messages), True

    def service(self) -> GmailSyncService:
        return GmailSyncService(
            vector_store=self.store,
            embedding_service=self.embedding,
            metadata=self.state,
            fetcher=self.fetcher,
        )

    def test_initial_sync_is_idempotent_and_skips_unchanged_messages(self) -> None:
        sync = self.service()
        first = sync.sync("user-a")
        points_after_first = self.store.count(user_id="user-a")
        second = sync.sync("user-a")
        self.assertEqual(first.messages_new, 2)
        self.assertEqual(second.messages_unchanged, 2)
        self.assertEqual(second.messages_new, 0)
        self.assertEqual(self.store.count(user_id="user-a"), points_after_first)
        self.assertEqual(self.embedding.calls, first.chunks_created)

    def test_changed_message_replaces_only_its_chunks(self) -> None:
        sync = self.service()
        sync.sync("user-a")
        self.messages[0] = message("one", "Changed body.")
        updated = sync.sync("user-a")
        self.assertEqual(updated.messages_updated, 1)
        self.assertEqual(updated.messages_unchanged, 1)
        self.assertEqual(self.store.count(user_id="user-a"), 2)
        subjects = [item.chunk["message_id"] for item in self.store.scroll_chunks(user_id="user-a")]
        self.assertEqual(set(subjects), {"one", "two"})

    def test_complete_sync_deletes_only_missing_user_messages(self) -> None:
        sync = self.service()
        sync.sync("user-a")
        other = message("one", "Other user body.")
        self.store.index_messages([other], self.embedding, user_id="user-b")
        self.messages = [self.messages[0]]
        result = sync.sync("user-a")
        self.assertEqual(result.messages_deleted, 1)
        self.assertEqual(self.store.count(user_id="user-a"), 1)
        self.assertEqual(self.store.count(user_id="user-b"), 1)

    def test_partial_sync_does_not_delete_unseen_messages(self) -> None:
        sync = self.service()
        sync.sync("user-a")
        sync.fetcher = lambda **kwargs: ([self.messages[0]], False)
        result = sync.sync("user-a")
        self.assertEqual(result.messages_deleted, 0)
        self.assertEqual(self.store.count(user_id="user-a"), 2)


class GmailPaginationTests(unittest.TestCase):
    class Messages:
        def __init__(self, pages):
            self.pages = pages
            self.requested_tokens = []

        def list(self, *, userId, maxResults, pageToken=None):
            self.requested_tokens.append(pageToken)
            index = 0 if pageToken is None else int(pageToken)
            listing = self.pages[index]
            return type("Request", (), {"execute": lambda self: listing})()

        def get(self, *, userId, id, format):
            payload = {"id": id, "threadId": f"thread-{id}", "payload": {"headers": [], "body": {}}}
            return type("Request", (), {"execute": lambda self: payload})()

    class Service:
        def __init__(self, pages):
            self.messages_api = GmailPaginationTests.Messages(pages)

        def users(self):
            messages_api = self.messages_api
            return type("Users", (), {"messages": lambda self: messages_api})()

    def test_pages_follow_tokens_and_stop_at_max(self) -> None:
        pages = [
            {"messages": [{"id": "one"}], "nextPageToken": "1"},
            {"messages": [{"id": "two"}], "nextPageToken": "2"},
            {"messages": [{"id": "three"}]},
        ]
        service = self.Service(pages)
        messages, complete = fetch_messages_paginated(max_messages=3, page_size=1, service=service)
        self.assertEqual([item.message_id for item in messages], ["one", "two", "three"])
        self.assertTrue(complete)
        self.assertEqual(service.messages_api.requested_tokens, [None, "1", "2"])


if __name__ == "__main__":
    unittest.main()
