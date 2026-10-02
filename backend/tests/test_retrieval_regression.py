import unittest

from qdrant_client import QdrantClient

from backend.app.chunking import EmailChunk
from backend.app.embeddings import EmbeddedChunk
from backend.app.query_service import evaluate_gmail_query
from backend.app.rag_service import generate_gmail_answer
from backend.app.vector_store import QdrantVectorStore


def fixture_chunk(
    chunk_id: str,
    subject: str,
    thread_id: str,
    *,
    sender: str | None = None,
    timestamp: str = "2026-09-05T12:00:00+00:00",
    text: str | None = None,
) -> EmailChunk:
    return EmailChunk(
        chunk_id=chunk_id,
        message_id=f"message-{chunk_id}",
        thread_id=thread_id,
        subject=subject,
        sender=sender or ("university@example.com" if "semester" in subject.lower() else "bank@example.com"),
        recipients=["student@example.com"],
        timestamp=timestamp,
        date=timestamp,
        labels=["INBOX"],
        chunk_index=0,
        total_chunks=1,
        text=text or subject,
    )


class FixedEmbeddingService:
    def embed_text(self, query: str) -> list[float]:
        return [1.0, 0.0, 0.0]


class RetrievalRegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = QdrantVectorStore(
            client=QdrantClient(location=":memory:"),
            collection_name="retrieval_regression",
            vector_size=3,
        )
        university = fixture_chunk("university", "Third semester fee notice", "university-thread")
        bank = fixture_chunk("bank", "Statement of your Account", "bank-thread")
        self.store.upsert(
            [
                EmbeddedChunk(university, [1.0, 0.0, 0.0]),
                EmbeddedChunk(bank, [0.2, 0.9, 0.0]),
            ],
            user_id="personal-gmail",
        )

    def test_equivalent_fee_queries_find_relevant_fixture(self) -> None:
        queries = (
            "semester fee",
            "third semester fee",
            "How much do I have to pay for semester 3?",
            "What was my university fee?",
            "Tell me about my semester fee.",
        )
        for query in queries:
            result = evaluate_gmail_query(
                "personal-gmail",
                query,
                vector_store=self.store,
                embedding_service=FixedEmbeddingService(),
                evidence_threshold=0.62,
            )
            self.assertEqual(result.decision, "allow", query)
            self.assertEqual(result.retrieved_chunks[0].chunk["subject"], "Third semester fee notice")

    def test_bank_fixture_does_not_dominate_relevant_result(self) -> None:
        result = evaluate_gmail_query(
            "personal-gmail",
            "MCA semester fee",
            vector_store=self.store,
            embedding_service=FixedEmbeddingService(),
        )
        self.assertEqual(result.retrieved_chunks[0].chunk["thread_id"], "university-thread")

    def test_unrelated_and_no_evidence_still_skip_groq(self) -> None:
        class CountingGroq:
            calls = 0

            def generate(self, **kwargs):
                self.calls += 1
                return "should not run"

        groq = CountingGroq()
        unrelated = generate_gmail_answer(
            "personal-gmail",
            "What is the capital of France?",
            evaluator=lambda user_id, query: evaluate_gmail_query(
                user_id,
                query,
                vector_store=self.store,
                embedding_service=FixedEmbeddingService(),
            ),
            groq_service=groq,
        )
        self.assertEqual(unrelated.reason, "unrelated_to_gmail")
        self.assertEqual(groq.calls, 0)

    def test_exact_lexical_terms_recover_when_dense_similarity_is_weak(self) -> None:
        instagram = fixture_chunk(
            "instagram", "Instagram login alert", "instagram-thread",
            sender="Instagram <security@mail.instagram.com>",
            text="Instagram login notification",
        )
        bank = fixture_chunk("bank", "Account statement", "bank-thread")
        store = QdrantVectorStore(
            client=QdrantClient(location=":memory:"), collection_name="lexical", vector_size=3
        )
        store.upsert(
            [
                EmbeddedChunk(instagram, [0.0, 1.0, 0.0]),
                EmbeddedChunk(bank, [1.0, 0.0, 0.0]),
            ],
            user_id="personal-gmail",
        )
        result = evaluate_gmail_query(
            "personal-gmail", "recent mail for instagarm login", vector_store=store,
            embedding_service=FixedEmbeddingService(), evidence_threshold=0.62,
        )
        self.assertEqual(result.decision, "allow")
        self.assertEqual(result.retrieved_chunks[0].chunk["subject"], "Instagram login alert")
        self.assertIn("instagram", " ".join(result.query_variants))

        class CountingGroq:
            calls = 0

            def generate(self, **kwargs):
                self.calls += 1
                return "Grounded Instagram result."

        groq = CountingGroq()
        grounded = generate_gmail_answer(
            "personal-gmail",
            "recent mail for IG login",
            evaluator=lambda user_id, query: evaluate_gmail_query(
                user_id,
                query,
                vector_store=store,
                embedding_service=FixedEmbeddingService(),
                evidence_threshold=0.62,
            ),
            groq_service=groq,
        )
        self.assertEqual(grounded.decision, "allow")
        self.assertEqual(groq.calls, 1)
        self.assertEqual(len(grounded.sources), 1)

    def test_missing_named_entity_evidence_fails_closed(self) -> None:
        store = QdrantVectorStore(
            client=QdrantClient(location=":memory:"), collection_name="missing-entity", vector_size=3
        )
        bank = fixture_chunk("bank", "Security alert", "bank-thread")
        store.upsert([EmbeddedChunk(bank, [1.0, 0.0, 0.0])], user_id="personal-gmail")
        result = evaluate_gmail_query(
            "personal-gmail", "recent mail for instagram login", vector_store=store,
            embedding_service=FixedEmbeddingService(), evidence_threshold=0.62,
        )
        self.assertEqual(result.reason, "insufficient_gmail_evidence")
        self.assertFalse(result.evidence_sufficient)

    def test_latest_query_prefers_newest_relevant_message(self) -> None:
        older = fixture_chunk(
            "older", "Security alert", "older-thread", sender="Google <security@google.com>",
            timestamp="2026-09-01T12:00:00+00:00",
        )
        newer = fixture_chunk(
            "newer", "Security alert", "newer-thread", sender="Google <security@google.com>",
            timestamp="2026-09-05T12:00:00+00:00",
        )
        store = QdrantVectorStore(
            client=QdrantClient(location=":memory:"), collection_name="temporal", vector_size=3
        )
        store.upsert(
            [EmbeddedChunk(older, [1.0, 0.0, 0.0]), EmbeddedChunk(newer, [1.0, 0.0, 0.0])],
            user_id="personal-gmail",
        )
        result = evaluate_gmail_query(
            "personal-gmail", "latest security email", vector_store=store,
            embedding_service=FixedEmbeddingService(), evidence_threshold=0.62,
        )
        self.assertEqual(result.retrieved_chunks[0].chunk["chunk_id"], "newer")

    def test_subject_match_beats_body_only_overlap(self) -> None:
        subject_match = fixture_chunk(
            "subject-match", "Instagram login detected", "subject-thread",
            sender="Instagram <security@mail.instagram.com>",
            text="A notification was sent.",
        )
        body_match = fixture_chunk(
            "body-match", "Google Security Alert", "body-thread",
            sender="Google <security@google.com>",
            text="Instagram login information was mentioned in this message.",
        )
        store = QdrantVectorStore(
            client=QdrantClient(location=":memory:"), collection_name="metadata", vector_size=3
        )
        store.upsert(
            [
                EmbeddedChunk(subject_match, [1.0, 0.0, 0.0]),
                EmbeddedChunk(body_match, [1.0, 0.0, 0.0]),
            ],
            user_id="personal-gmail",
        )
        result = evaluate_gmail_query(
            "personal-gmail", "recent Instagram login", vector_store=store,
            embedding_service=FixedEmbeddingService(), evidence_threshold=0.62,
        )
        self.assertEqual(result.decision, "allow")
        self.assertEqual(result.retrieved_chunks[0].chunk["chunk_id"], "subject-match")

    def test_chunk_count_does_not_outvote_stronger_message(self) -> None:
        repeated = [
            fixture_chunk(
                f"repeated-{index}", "Security alert", "repeated-thread",
                timestamp="2026-09-01T12:00:00+00:00",
            )
            for index in range(5)
        ]
        distinct = fixture_chunk(
            "distinct", "Security alert", "distinct-thread",
            sender="Google <security@google.com>",
            timestamp="2026-09-05T12:00:00+00:00",
        )
        store = QdrantVectorStore(
            client=QdrantClient(location=":memory:"), collection_name="message-ranking", vector_size=3
        )
        store.upsert(
            [
                *[EmbeddedChunk(item, [1.0, 0.0, 0.0]) for item in repeated],
                EmbeddedChunk(distinct, [1.0, 0.0, 0.0]),
            ],
            user_id="personal-gmail",
        )
        result = evaluate_gmail_query(
            "personal-gmail", "latest security email", vector_store=store,
            embedding_service=FixedEmbeddingService(), evidence_threshold=0.62,
        )
        self.assertEqual(result.retrieved_chunks[0].chunk["chunk_id"], "distinct")

    def test_payload_scan_is_user_scoped(self) -> None:
        store = QdrantVectorStore(
            client=QdrantClient(location=":memory:"), collection_name="scope", vector_size=3
        )
        item = fixture_chunk("one", "Instagram login alert", "thread")
        store.upsert([EmbeddedChunk(item, [1.0, 0.0, 0.0])], user_id="user-a")
        store.upsert([EmbeddedChunk(item, [1.0, 0.0, 0.0])], user_id="user-b")
        self.assertEqual(len(store.scroll_chunks(user_id="user-a")), 1)
        self.assertEqual(store.scroll_chunks(user_id="user-a")[0].chunk["user_id"], "user-a")


if __name__ == "__main__":
    unittest.main()