import unittest

from backend.app.chunking import EmailChunk
from backend.app.models import NormalizedGmailMessage
from backend.app.query_planner import plan_search
from backend.app.query_service import evaluate_gmail_query
from backend.app.rag_service import NO_EVIDENCE_ANSWER, generate_gmail_answer
from backend.app.vector_store import QdrantVectorStore
from qdrant_client import QdrantClient


class BrokenEmbedding:
    def embed_text(self, text):
        raise AssertionError("structured search must not embed")


class NoSearchStore:
    vector_size = 3

    def search(self, *args, **kwargs):
        raise AssertionError("live structured results should not call Qdrant search")


def message(message_id: str, internal_date: int, *, labels=None) -> NormalizedGmailMessage:
    return NormalizedGmailMessage(
        message_id=message_id,
        thread_id=f"thread-{message_id}",
        subject="Unread receipt",
        sender="Billing <billing@example.com>",
        recipients=["person@example.com"],
        timestamp="2026-09-02T12:00:00+00:00",
        date="Tue, 02 Sep 2026 12:00:00 +0000",
        body="Your payment receipt is attached.",
        snippet="Your payment receipt is attached.",
        labels=labels or ["INBOX", "UNREAD"],
        internal_date=internal_date,
    )


class StructuredRetrievalTests(unittest.TestCase):
    def test_structured_match_is_not_returned_as_an_insufficient_answer(self):
        from backend.app.query_service import GmailQueryEvaluation
        from backend.app.vector_store import RetrievedChunk

        chunk = EmailChunk(
            chunk_id="spam-1:0", message_id="spam-1", thread_id="thread-spam-1",
            subject="Product update", sender="Google Workspace Team <team@google.com>",
            recipients=["person@example.com"], timestamp="2026-09-08T10:00:00+00:00",
            date="Mon, 08 Sep 2026 10:00:00 +0000", labels=["SPAM"],
            chunk_index=0, total_chunks=1, internal_date=1788861600000,
            text="Subject: Product update\n\nYou now have exclusive access to AI features.",
        )
        evaluation = GmailQueryEvaluation(
            query="show messages in a mailbox state", relevant_to_gmail=True,
            decision="allow", reason="sufficient_gmail_evidence",
            retrieved_chunks=(RetrievedChunk(chunk, 1.0),),
            search_plan=plan_search("show messages in spam"),
        )

        class IncorrectlyCautiousGroq:
            def generate(self, **kwargs):
                return NO_EVIDENCE_ANSWER

        result = generate_gmail_answer(
            "user-1", "show messages in a mailbox state",
            evaluator=lambda *_args, **_kwargs: evaluation,
            groq_service=IncorrectlyCautiousGroq(),
        )
        self.assertEqual(result.decision, "allow")
        self.assertNotEqual(result.answer, NO_EVIDENCE_ANSWER)
        self.assertIn("Product update", result.answer or "")

    def test_structured_result_metadata_is_not_reduced_to_a_score(self):
        from unittest.mock import patch

        live = message("gmail-message-1", 1770000000000)
        with patch("backend.app.query_service.search_messages", return_value=([live], True)):
            result = evaluate_gmail_query(
                "user-1",
                "show unread emails",
                vector_store=NoSearchStore(),
                embedding_service=BrokenEmbedding(),
                gmail_service=object(),
            )
        self.assertEqual(result.decision, "allow")
        evidence = result.retrieved_chunks[0].chunk
        self.assertEqual(evidence["message_id"], "gmail-message-1")
        self.assertEqual(evidence["thread_id"], "thread-gmail-message-1")
        self.assertEqual(evidence["subject"], "Unread receipt")
        self.assertEqual(evidence["sender"], "Billing <billing@example.com>")
        self.assertEqual(evidence["internal_date"], 1770000000000)
        self.assertEqual(evidence["labels"], ["INBOX", "UNREAD"])
        self.assertIn("payment receipt", evidence["text"])

    def test_date_filter_uses_internal_date_not_display_date(self):
        class Embeddings:
            def embed_chunks(self, chunks):
                from backend.app.embeddings import EmbeddedChunk
                return [EmbeddedChunk(chunk, [1.0, 0.0, 0.0]) for chunk in chunks]

            def embed_text(self, text):
                return [1.0, 0.0, 0.0]

        store = QdrantVectorStore(
            client=QdrantClient(location=":memory:"), collection_name="date-filter", vector_size=3
        )
        old = message("old", 1000, labels=["INBOX"])
        new = message("new", 3000, labels=["INBOX"])
        # Deliberately misleading display dates must not affect filtering.
        old = old.__class__(**{**old.__dict__, "date": "2099-01-01"})
        new = new.__class__(**{**new.__dict__, "date": "2000-01-01"})
        store.index_messages([old, new], Embeddings(), user_id="user-date")
        plan = plan_search("emails after:1970/01/01")
        self.assertEqual(plan.retrieval_mode, "structured")
        result = evaluate_gmail_query(
            "user-date", "emails after:1970/01/01", vector_store=store,
            embedding_service=BrokenEmbedding(),
        )
        self.assertEqual([item.chunk["message_id"] for item in result.retrieved_chunks], ["new", "old"])

    def test_planner_supports_recipient_subject_attachment_and_native_dates(self):
        plan = plan_search(
            'emails to:person@example.com subject:"receipt" with attachment after:2026/09/01 before:2026/10/01'
        )
        self.assertEqual(plan.recipient, "person@example.com")
        self.assertEqual(plan.subject, "receipt")
        self.assertTrue(plan.has_attachment)
        self.assertIn("to:person@example.com", plan.gmail_query)
        self.assertIn("has:attachment", plan.gmail_query)
        self.assertEqual(plan.internal_date_after, 1788220800000)
        self.assertEqual(plan.internal_date_before, 1790812800000)

    def test_explicit_entities_are_structured_constraints_and_match_message_content(self):
        plan = plan_search("recent Acme related mail")
        self.assertIn("acme", [entity.lower() for entity in plan.entities])
        self.assertTrue(plan.use_structured_search)
        self.assertIn("Acme", plan.gmail_query)
        self.assertEqual(plan.sort, "newest")

        class Embeddings:
            def embed_text(self, text):
                return [1.0, 0.0]

        class Store:
            vector_size = 2

            def search(self, vector, *, user_id, limit):
                return [
                    # Generic semantic match but missing the explicit entity.
                    type("Result", (), {"chunk": EmailChunk(
                        chunk_id="other", message_id="other", thread_id="other-thread",
                        subject="Bank statement", sender="Bank <notice@bank.test>", recipients=[],
                        timestamp="", date="", labels=["INBOX"], chunk_index=0, total_chunks=1,
                        internal_date=3000, text="Your banking update",
                    ), "score": 0.99})(),
                    type("Result", (), {"chunk": EmailChunk(
                        chunk_id="acme", message_id="acme", thread_id="acme-thread",
                        subject="Quarterly note", sender="Updates <notice@updates.test>", recipients=[],
                        timestamp="", date="", labels=["INBOX"], chunk_index=0, total_chunks=1,
                        internal_date=2000, text="Acme account update",
                    ), "score": 0.70})(),
                ]

        result = evaluate_gmail_query(
            "user-1", "Acme related mail", vector_store=Store(),
            embedding_service=Embeddings(), evidence_threshold=0.1,
        )
        self.assertEqual([item.chunk["message_id"] for item in result.retrieved_chunks], ["acme"])

    def test_live_structured_entity_match_survives_semantic_threshold(self):
        from unittest.mock import patch
        from backend.app.vector_store import RetrievedChunk

        live = message("acme-live", 2000)
        live = live.__class__(
            **{
                **live.__dict__,
                "subject": "Acme account update",
                "sender": "Acme <updates@acme.test>",
                "body": "Your Acme account update is available.",
            }
        )
        unrelated = EmailChunk(
            chunk_id="bank-neighbor", message_id="bank-neighbor", thread_id="bank-thread",
            subject="Bank statement", sender="Bank <notice@bank.test>", recipients=[],
            timestamp="", date="", labels=["INBOX"], chunk_index=0, total_chunks=1,
            internal_date=3000, text="Your banking update",
        )

        class Embeddings:
            def embed_text(self, text):
                return [1.0, 0.0]

        class Store:
            def search(self, vector, *, user_id, limit):
                return [RetrievedChunk(unrelated, 0.99)]

        with patch("backend.app.query_service.search_messages", return_value=([live], True)):
            result = evaluate_gmail_query(
                "user-1",
                "recent Acme related mail",
                vector_store=Store(),
                embedding_service=Embeddings(),
                evidence_threshold=0.995,
                gmail_service=object(),
            )

        self.assertEqual(result.decision, "allow")
        self.assertEqual(
            [item.chunk["message_id"] for item in result.retrieved_chunks],
            ["acme-live"],
        )


if __name__ == "__main__":
    unittest.main()
