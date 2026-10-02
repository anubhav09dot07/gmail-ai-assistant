import unittest

from backend.app.chunking import chunk_message
from backend.app.models import NormalizedGmailMessage


def message(body: str) -> NormalizedGmailMessage:
    return NormalizedGmailMessage(
        message_id="message-1",
        thread_id="thread-1",
        subject="Semester Fee Payment",
        sender="university@example.com",
        recipients=["student@example.com"],
        timestamp="2026-09-05T12:00:00+00:00",
        date="2026-09-05T12:00:00+00:00",
        body=body,
        snippet=body[:40],
        labels=["INBOX"],
    )


class ChunkingTests(unittest.TestCase):
    def test_short_email_creates_one_chunk_with_metadata(self) -> None:
        chunks = chunk_message(message("Your fee is 105000 rupees."), chunk_size=200)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["message_id"], "message-1")
        self.assertEqual(chunks[0]["thread_id"], "thread-1")
        self.assertIn("Subject: Semester Fee Payment", chunks[0].text)

    def test_long_email_creates_ordered_chunks(self) -> None:
        body = " ".join(["Important deadline 10 September."] * 30)
        chunks = chunk_message(message(body), chunk_size=180)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(
            [chunk["chunk_index"] for chunk in chunks], list(range(len(chunks)))
        )
        self.assertTrue(all(chunk["total_chunks"] == len(chunks) for chunk in chunks))

    def test_overlap_and_unique_ids(self) -> None:
        body = " ".join(f"fact-{index}" for index in range(60))
        chunks = chunk_message(message(body), chunk_size=160, overlap=30)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk.text) <= 160 for chunk in chunks))
        self.assertTrue(
            any(
                set(first.text.split()) & set(second.text.split())
                for first, second in zip(chunks, chunks[1:])
            )
        )
        self.assertEqual(len({chunk["chunk_id"] for chunk in chunks}), len(chunks))

    def test_empty_email_creates_no_chunks(self) -> None:
        self.assertEqual(chunk_message(message("   ")), [])


if __name__ == "__main__":
    unittest.main()