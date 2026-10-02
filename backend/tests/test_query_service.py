import inspect
import unittest

from backend.app.chunking import EmailChunk
from backend.app.embeddings import EmbeddingService
from backend.app.query_service import evaluate_gmail_query
from backend.app.vector_store import RetrievedChunk


def retrieved(subject: str, score: float, thread_id: str = "thread-1") -> RetrievedChunk:
    return RetrievedChunk(
        chunk=EmailChunk(
            chunk_id=subject, message_id=f"message-{subject}", thread_id=thread_id,
            subject=subject, sender="sender@example.com", recipients=[],
            timestamp="2026-09-05T12:00:00+00:00", date="2026-09-05T12:00:00+00:00",
            labels=["INBOX"], chunk_index=0, total_chunks=1, text="private body",
        ),
        score=score,
    )


class FakeEmbeddingService:
    def embed_text(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]


class FakeStore:
    def __init__(self, results: list[RetrievedChunk]) -> None:
        self.results = results
        self.calls = 0
        self.user_ids: list[str] = []

    def search(self, vector, *, user_id: str, limit: int):
        self.calls += 1
        self.user_ids.append(user_id)
        return self.results[:limit]


class QueryServiceTests(unittest.TestCase):
    def test_security_queries_are_recognized_as_gmail_related(self) -> None:
        queries = (
            "What was the last security message?",
            "What was my latest security alert?",
            "Did I receive a security notification?",
            "Show me my recent login alerts.",
            "When was the last sign-in alert?",
            "Was there a new sign-in?",
            "Was there any suspicious account activity email?",
        )
        for query in queries:
            with self.subTest(query=query):
                store = FakeStore([])
                result = evaluate_gmail_query(
                    "user-1", query, vector_store=store,
                    embedding_service=FakeEmbeddingService(), evidence_threshold=0.62,
                )
                self.assertTrue(result.relevant_to_gmail)
                self.assertEqual(result.reason, "insufficient_gmail_evidence")
                self.assertEqual(store.calls, 1)

    def test_other_gmail_topic_queries_are_recognized(self) -> None:
        queries = (
            "Which receipts did I receive?",
            "What is my latest subscription payment?",
            "Show me my travel booking documents.",
        )
        for query in queries:
            with self.subTest(query=query):
                result = evaluate_gmail_query(
                    "user-1", query, vector_store=FakeStore([]),
                    embedding_service=FakeEmbeddingService(),
                )
                self.assertTrue(result.relevant_to_gmail)

    def test_unrelated_questions_remain_rejected_before_qdrant(self) -> None:
        queries = (
            "What is the capital of France?",
            "How does quantum mechanics work?",
            "Write a Python program.",
            "Who is the president of India?",
        )
        for query in queries:
            with self.subTest(query=query):
                store = FakeStore([retrieved("private", 0.99)])
                result = evaluate_gmail_query(
                    "user-1", query, vector_store=store,
                    embedding_service=FakeEmbeddingService(),
                )
                self.assertFalse(result.relevant_to_gmail)
                self.assertEqual(result.reason, "unrelated_to_gmail")
                self.assertEqual(store.calls, 0)

    def test_unrelated_query_rejects_before_qdrant(self) -> None:
        store = FakeStore([retrieved("private", 0.99)])
        result = evaluate_gmail_query(
            "user-1", "What is the capital of France?", vector_store=store,
            embedding_service=FakeEmbeddingService()
        )
        self.assertFalse(result.relevant_to_gmail)
        self.assertEqual(result.reason, "unrelated_to_gmail")
        self.assertEqual(store.calls, 0)

    def test_gmail_related_query_with_no_evidence_rejects(self) -> None:
        result = evaluate_gmail_query(
            "user-1", "What is my passport number?", vector_store=FakeStore([]),
            embedding_service=FakeEmbeddingService(), evidence_threshold=0.62
        )
        self.assertTrue(result.relevant_to_gmail)
        self.assertEqual(result.reason, "insufficient_gmail_evidence")
        self.assertEqual(result.decision, "reject")

    def test_strong_evidence_allows_query(self) -> None:
        store = FakeStore([retrieved("Semester fee", 0.81)])
        result = evaluate_gmail_query(
            "user-1", "What did my university email say about the semester fee?",
            vector_store=store, embedding_service=FakeEmbeddingService()
        )
        self.assertEqual(result.decision, "allow")
        self.assertEqual(result.reason, "sufficient_gmail_evidence")

    def test_ambiguous_decision_question_is_plausibly_gmail(self) -> None:
        store = FakeStore([retrieved("Decision", 0.70)])
        result = evaluate_gmail_query(
            "user-1", "What did they finally decide?", vector_store=store,
            embedding_service=FakeEmbeddingService()
        )
        self.assertTrue(result.relevant_to_gmail)

    def test_low_score_rejects_and_threshold_is_configurable(self) -> None:
        store = FakeStore([retrieved("Nearest", 0.51)])
        result = evaluate_gmail_query(
            "user-1", "What is my bank statement amount?", vector_store=store,
            embedding_service=FakeEmbeddingService(), evidence_threshold=0.60
        )
        self.assertEqual(result.decision, "reject")
        self.assertEqual(result.evidence_threshold, 0.60)

    def test_user_isolation_is_forwarded_to_qdrant(self) -> None:
        store = FakeStore([retrieved("Fee", 0.80)])
        evaluate_gmail_query(
            "user-a", "What did my email say about the fee?", vector_store=store,
            embedding_service=FakeEmbeddingService()
        )
        self.assertEqual(store.user_ids, ["user-a"])

    def test_empty_query_and_invalid_user_reject(self) -> None:
        store = FakeStore([])
        empty = evaluate_gmail_query(
            "user-1", "  ", vector_store=store, embedding_service=FakeEmbeddingService()
        )
        invalid = evaluate_gmail_query(
            "", "What did my email say?", vector_store=store,
            embedding_service=FakeEmbeddingService()
        )
        self.assertEqual(empty.reason, "empty_query")
        self.assertEqual(invalid.reason, "invalid_user_id")
        self.assertEqual(store.calls, 0)

    def test_thread_chunks_are_grouped(self) -> None:
        store = FakeStore([
            retrieved("One", 0.80, "thread-a"),
            retrieved("Two", 0.79, "thread-a"),
            retrieved("Three", 0.78, "thread-b"),
        ])
        result = evaluate_gmail_query(
            "user-1", "What did we decide in the email thread?", vector_store=store,
            embedding_service=FakeEmbeddingService(), limit=3
        )
        self.assertEqual(len(result.thread_groups["thread-a"]), 2)
        self.assertEqual(len(result.thread_groups["thread-b"]), 1)

    def test_public_summary_does_not_expose_bodies(self) -> None:
        result = evaluate_gmail_query(
            "user-1", "What did my email say about the fee?",
            vector_store=FakeStore([retrieved("Fee", 0.80)]),
            embedding_service=FakeEmbeddingService()
        )
        self.assertNotIn("private body", str(result.public_summary()))

    def test_embedding_failure_rejects_closed(self) -> None:
        class BrokenEmbedding:
            def embed_text(self, text: str):
                raise RuntimeError("model unavailable")

        result = evaluate_gmail_query(
            "user-1", "What did my email say?", vector_store=FakeStore([]),
            embedding_service=BrokenEmbedding()
        )
        self.assertEqual(result.reason, "embedding_unavailable")

    def test_qdrant_failure_rejects_closed(self) -> None:
        class BrokenStore:
            def search(self, vector, *, user_id: str, limit: int):
                raise RuntimeError("qdrant unavailable")

        result = evaluate_gmail_query(
            "user-1", "What did my email say?", vector_store=BrokenStore(),
            embedding_service=FakeEmbeddingService()
        )
        self.assertEqual(result.decision, "reject")
        self.assertEqual(result.reason, "qdrant_unavailable")

    def test_gate_has_no_groq_generation_dependency(self) -> None:
        from backend.app import query_service

        self.assertNotIn("groq", inspect.getsource(query_service).lower())


if __name__ == "__main__":
    unittest.main()