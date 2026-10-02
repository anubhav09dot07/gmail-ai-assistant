import unittest
from backend.app.chunking import EmailChunk
from backend.app.gmail import normalize_message, search_messages
from backend.app.models import NormalizedGmailMessage
from backend.app.query_planner import plan_search
from backend.app.query_service import evaluate_gmail_query
from backend.app.rag_service import RAGSource, generate_gmail_answer
from backend.app.vector_store import QdrantVectorStore, RetrievedChunk
from qdrant_client import QdrantClient


class DeterministicEmbeddingService:
    """Mock embedding service that produces high similarity for matching keywords."""

    def __init__(self) -> None:
        self.dimension = 8

    def _vector(self, text: str) -> list[float]:
        t = text.lower()
        vec = [0.0] * self.dimension
        if "kyc" in t:
            vec[0] = 1.0
        if "mca" in t or "fee" in t:
            vec[1] = 1.0
        if "deadline" in t or "project" in t:
            vec[2] = 1.0
        if "vacation" in t:
            vec[3] = 1.0
        if "security" in t or "login" in t:
            vec[4] = 1.0
        if "bank" in t or "statement" in t:
            vec[5] = 1.0
        norm = sum(x * x for x in vec) ** 0.5
        if norm > 0:
            return [x / norm for x in vec]
        return [0.1] * self.dimension

    def embed_text(self, text: str) -> list[float]:
        return self._vector(text)

    def embed_chunks(self, chunks):
        from backend.app.embeddings import EmbeddedChunk

        return [
            EmbeddedChunk(chunk, self._vector(chunk.text)) for chunk in chunks
        ]


def create_test_message(
    *,
    message_id: str,
    subject: str,
    body: str,
    sender: str,
    internal_date: int,
    labels: list[str] | None = None,
    thread_id: str = "",
) -> NormalizedGmailMessage:
    return NormalizedGmailMessage(
        message_id=message_id,
        thread_id=thread_id or f"thread-{message_id}",
        subject=subject,
        sender=sender,
        recipients=["user@example.com"],
        timestamp="2026-09-05T12:00:00+00:00",
        date="2026-09-05T12:00:00+00:00",
        body=body,
        snippet=body[:40],
        labels=labels or ["INBOX"],
        internal_date=internal_date,
    )


class HybridRetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.embedding = DeterministicEmbeddingService()
        self.store = QdrantVectorStore(
            client=QdrantClient(location=":memory:"),
            collection_name="hybrid-test-collection",
            vector_size=8,
            batch_size=10,
        )

    def test_query_planner_extracts_orthogonal_facets(self) -> None:
        # Hybrid query: Binance KYC
        plan1 = plan_search("latest email from Binance about KYC")
        self.assertTrue(plan1.is_gmail_related)
        self.assertEqual(plan1.sender, "Binance")
        self.assertEqual(plan1.sort, "newest")
        self.assertIn("kyc", plan1.semantic_query.lower())
        self.assertTrue(plan1.use_structured_search)
        self.assertTrue(plan1.use_semantic_search)

        # Hybrid query: College MCA fees
        plan2 = plan_search("emails from my college about MCA fees")
        self.assertTrue(plan2.is_gmail_related)
        self.assertEqual(plan2.sender, "college")
        self.assertIn("mca", plan2.semantic_query.lower())
        self.assertTrue(plan2.use_structured_search)
        self.assertTrue(plan2.use_semantic_search)

        # Pure structured query: Unread
        plan3 = plan_search("unread emails")
        self.assertTrue(plan3.is_gmail_related)
        self.assertIn("UNREAD", plan3.labels)
        self.assertTrue(plan3.use_structured_search)
        self.assertFalse(plan3.use_semantic_search)

        # Pure structured query: Spam
        plan4 = plan_search("emails in spam")
        self.assertTrue(plan4.is_gmail_related)
        self.assertIn("SPAM", plan4.labels)
        self.assertTrue(plan4.include_spam_trash)
        self.assertTrue(plan4.use_structured_search)
        self.assertFalse(plan4.use_semantic_search)

        # Pure structured query: Latest from Binance
        plan5 = plan_search("latest email from Binance")
        self.assertTrue(plan5.is_gmail_related)
        self.assertEqual(plan5.sender, "Binance")
        self.assertEqual(plan5.sort, "newest")
        self.assertTrue(plan5.use_structured_search)
        self.assertFalse(plan5.use_semantic_search)

    def test_hybrid_binance_kyc_hard_filters_and_orders_by_newest_internal_date(
        self,
    ) -> None:
        """For 'latest email from Binance about KYC':

        1. Binance + KYC (older internal_date=1000)
        2. Binance + KYC (newer internal_date=2000) -> MUST BE THE WINNER
        3. Binance + Password Reset (no KYC) -> rejected by semantic evidence
        4. Coinbase + KYC (newer internal_date=3000) -> rejected by sender hard
        filter
        """
        messages = [
            create_test_message(
                message_id="binance-kyc-old",
                subject="Binance Verification",
                body="Please complete your KYC identity verification today.",
                sender="Binance Support <support@binance.com>",
                internal_date=1000,
            ),
            create_test_message(
                message_id="binance-kyc-new",
                subject="Binance KYC Urgent Update",
                body="Your KYC identity verification documents were reviewed and approved.",
                sender="Binance Notifications <notifications@binance.com>",
                internal_date=2000,
            ),
            create_test_message(
                message_id="binance-reset",
                subject="Binance Password Reset",
                body="You requested a password reset for your Binance account.",
                sender="Binance Security <security@binance.com>",
                internal_date=2500,
            ),
            create_test_message(
                message_id="coinbase-kyc",
                subject="Coinbase KYC Verification",
                body="Please submit your KYC documents to Coinbase.",
                sender="Coinbase Compliance <compliance@coinbase.com>",
                internal_date=3000,
            ),
        ]
        self.store.index_messages(
            messages, self.embedding, user_id="user-crypto"
        )

        result = evaluate_gmail_query(
            "user-crypto",
            "latest email from Binance about KYC",
            vector_store=self.store,
            embedding_service=self.embedding,
            evidence_threshold=0.62,
        )
        self.assertEqual(result.decision, "allow")
        self.assertTrue(len(result.retrieved_chunks) >= 1)

        # Winner must be binance-kyc-new
        top_chunk = result.retrieved_chunks[0]
        self.assertEqual(top_chunk.chunk["message_id"], "binance-kyc-new")
        self.assertIn("binance", top_chunk.chunk["sender"].lower())
        self.assertEqual(top_chunk.chunk["internal_date"], 2000)

        # Verify Coinbase was strictly excluded by hard filter
        retrieved_ids = {c.chunk["message_id"] for c in result.retrieved_chunks}
        self.assertNotIn("coinbase-kyc", retrieved_ids)
        # Verify Password reset (no KYC) was excluded by semantic evidence
        self.assertNotIn("binance-reset", retrieved_ids)

    def test_hybrid_college_mca_fees_hard_filters_and_matches_topic(
        self,
    ) -> None:
        """For 'emails from my college about MCA fees':

        - College email about MCA fees -> ALLOW
        - Bank email about fee payment -> REJECT (sender is Bank, not College)
        - College email about sports event -> REJECT (no MCA fees topic)
        """
        messages = [
            create_test_message(
                message_id="college-mca",
                subject="College Semester MCA Fee Notice",
                body="The deadline for paying the MCA semester fees is next Monday.",
                sender="College Administration <admin@college.edu>",
                internal_date=1000,
            ),
            create_test_message(
                message_id="bank-fee",
                subject="Bank Receipt for Fee Payment",
                body="Payment of university MCA fee received successfully by the bank.",
                sender="HDFC Bank <alerts@hdfcbank.com>",
                internal_date=1200,
            ),
            create_test_message(
                message_id="college-sports",
                subject="College Sports Meet",
                body="Annual college sports day will be held this Saturday on campus.",
                sender="College Sports Committee <sports@college.edu>",
                internal_date=1400,
            ),
        ]
        self.store.index_messages(messages, self.embedding, user_id="user-edu")

        result = evaluate_gmail_query(
            "user-edu",
            "emails from my college about MCA fees",
            vector_store=self.store,
            embedding_service=self.embedding,
            evidence_threshold=0.62,
        )
        self.assertEqual(result.decision, "allow")
        retrieved_ids = [c.chunk["message_id"] for c in result.retrieved_chunks]
        self.assertEqual(retrieved_ids, ["college-mca"])

    def test_most_recent_project_deadline_orders_by_internal_date(self) -> None:
        """For 'most recent email mentioning the project deadline':

        Ordering must be strictly descending by internal_date.
        """
        messages = [
            create_test_message(
                message_id="deadline-old",
                subject="Project Deadline",
                body="Initial project deadline is set for March 15.",
                sender="lead@work.com",
                internal_date=5000,
            ),
            create_test_message(
                message_id="deadline-new",
                subject="Project Deadline Extension",
                body="The project deadline has been officially extended to March 30.",
                sender="manager@work.com",
                internal_date=9000,
            ),
        ]
        self.store.index_messages(messages, self.embedding, user_id="user-work")

        result = evaluate_gmail_query(
            "user-work",
            "most recent email mentioning the project deadline",
            vector_store=self.store,
            embedding_service=self.embedding,
            evidence_threshold=0.62,
        )
        self.assertEqual(result.decision, "allow")
        self.assertEqual(
            result.retrieved_chunks[0].chunk["message_id"], "deadline-new"
        )
        self.assertEqual(
            result.retrieved_chunks[0].chunk["internal_date"], 9000
        )

    def test_unread_emails_hard_filter_strictly_enforces_unread_label(
        self,
    ) -> None:
        messages = [
            create_test_message(
                message_id="read-1",
                subject="Read Newsletter",
                body="General news update.",
                sender="news@example.com",
                internal_date=100,
                labels=["INBOX"],
            ),
            create_test_message(
                message_id="unread-1",
                subject="Unread Alert",
                body="You have a new message waiting.",
                sender="service@example.com",
                internal_date=200,
                labels=["INBOX", "UNREAD"],
            ),
            create_test_message(
                message_id="unread-2",
                subject="Another Unread Message",
                body="Second unread message.",
                sender="boss@example.com",
                internal_date=300,
                labels=["INBOX", "UNREAD"],
            ),
        ]
        self.store.index_messages(
            messages, self.embedding, user_id="user-unread"
        )

        result = evaluate_gmail_query(
            "user-unread",
            "unread emails",
            vector_store=self.store,
            embedding_service=self.embedding,
        )
        self.assertEqual(result.decision, "allow")
        for chunk in result.retrieved_chunks:
            self.assertIn("UNREAD", chunk.chunk["labels"])
        retrieved_ids = {c.chunk["message_id"] for c in result.retrieved_chunks}
        self.assertNotIn("read-1", retrieved_ids)
        self.assertEqual(retrieved_ids, {"unread-1", "unread-2"})

    def test_spam_emails_hard_filter_strictly_enforces_spam_label(self) -> None:
        messages = [
            create_test_message(
                message_id="inbox-1",
                subject="Normal Email",
                body="Normal inbox email.",
                sender="friend@example.com",
                internal_date=100,
                labels=["INBOX"],
            ),
            create_test_message(
                message_id="spam-1",
                subject="You Won A Lottery",
                body="Claim your millions today!",
                sender="lottery@scam.org",
                internal_date=200,
                labels=["SPAM"],
            ),
        ]
        self.store.index_messages(messages, self.embedding, user_id="user-spam")

        # 1. Query specifically for spam
        spam_result = evaluate_gmail_query(
            "user-spam",
            "emails in spam",
            vector_store=self.store,
            embedding_service=self.embedding,
        )
        self.assertEqual(spam_result.decision, "allow")
        self.assertEqual(len(spam_result.retrieved_chunks), 1)
        self.assertEqual(
            spam_result.retrieved_chunks[0].chunk["message_id"], "spam-1"
        )
        self.assertIn("SPAM", spam_result.retrieved_chunks[0].chunk["labels"])

        # 2. General query must NOT return spam
        normal_result = evaluate_gmail_query(
            "user-spam",
            "latest email",
            vector_store=self.store,
            embedding_service=self.embedding,
        )
        self.assertEqual(normal_result.decision, "allow")
        normal_ids = {c.chunk["message_id"] for c in normal_result.retrieved_chunks}
        self.assertNotIn("spam-1", normal_ids)
        self.assertIn("inbox-1", normal_ids)

    def test_latest_email_from_binance_orders_by_internal_date(self) -> None:
        messages = [
            create_test_message(
                message_id="binance-old",
                subject="Old Binance statement",
                body="Old report",
                sender="Binance <info@binance.com>",
                internal_date=1000,
            ),
            create_test_message(
                message_id="binance-new",
                subject="New Binance announcement",
                body="New report",
                sender="Binance <info@binance.com>",
                internal_date=5000,
            ),
            create_test_message(
                message_id="other-newest",
                subject="Newest from someone else",
                body="Other report",
                sender="Apple <info@apple.com>",
                internal_date=9000,
            ),
        ]
        self.store.index_messages(
            messages, self.embedding, user_id="user-binance"
        )

        result = evaluate_gmail_query(
            "user-binance",
            "latest email from Binance",
            vector_store=self.store,
            embedding_service=self.embedding,
        )
        self.assertEqual(result.decision, "allow")
        self.assertEqual(
            result.retrieved_chunks[0].chunk["message_id"], "binance-new"
        )
        self.assertEqual(
            result.retrieved_chunks[0].chunk["internal_date"], 5000
        )

    def test_message_level_deduplication_keeps_single_representative_per_message(
        self,
    ) -> None:
        # Create a long message that splits into multiple chunks
        long_body = "The project deadline is critical.\n\n" * 20
        msg = create_test_message(
            message_id="multi-chunk-msg",
            subject="Project Deadline Final Warning",
            body=long_body,
            sender="boss@work.com",
            internal_date=3000,
        )
        self.store.index_messages([msg], self.embedding, user_id="user-chunks")

        result = evaluate_gmail_query(
            "user-chunks",
            "What is the project deadline?",
            vector_store=self.store,
            embedding_service=self.embedding,
        )
        self.assertEqual(result.decision, "allow")
        # Ensure only 1 chunk for multi-chunk-msg is in retrieved_chunks
        returned_ids = [
            c.chunk["message_id"] for c in result.retrieved_chunks
        ]
        self.assertEqual(returned_ids.count("multi-chunk-msg"), 1)

    def test_fail_closed_when_semantic_evidence_is_insufficient(self) -> None:
        msg = create_test_message(
            message_id="cooking-recipe",
            subject="Pasta Recipe",
            body="Boil water and cook pasta for 10 minutes.",
            sender="chef@recipe.com",
            internal_date=1000,
        )
        self.store.index_messages([msg], self.embedding, user_id="user-recipe")

        result = evaluate_gmail_query(
            "user-recipe",
            "What is my bank account balance?",
            vector_store=self.store,
            embedding_service=self.embedding,
            evidence_threshold=0.62,
        )
        self.assertEqual(result.decision, "reject")
        self.assertEqual(result.reason, "insufficient_gmail_evidence")

    def test_user_isolation_blocks_cross_tenant_access(self) -> None:
        msg_a = create_test_message(
            message_id="secret-a",
            subject="User A Secret Salary",
            body="Salary is 200000 dollars.",
            sender="hr@company.com",
            internal_date=1000,
        )
        msg_b = create_test_message(
            message_id="secret-b",
            subject="User B Secret Salary",
            body="Salary is 150000 dollars.",
            sender="hr@company.com",
            internal_date=1000,
        )
        self.store.index_messages([msg_a], self.embedding, user_id="tenant-a")
        self.store.index_messages([msg_b], self.embedding, user_id="tenant-b")

        result_a = evaluate_gmail_query(
            "tenant-a",
            "What is my salary?",
            vector_store=self.store,
            embedding_service=self.embedding,
        )
        result_b = evaluate_gmail_query(
            "tenant-b",
            "What is my salary?",
            vector_store=self.store,
            embedding_service=self.embedding,
        )

        for chunk in result_a.retrieved_chunks:
            self.assertEqual(chunk.chunk["message_id"], "secret-a")
            self.assertNotEqual(chunk.chunk["message_id"], "secret-b")

        for chunk in result_b.retrieved_chunks:
            self.assertEqual(chunk.chunk["message_id"], "secret-b")
            self.assertNotEqual(chunk.chunk["message_id"], "secret-a")

    def test_normalize_message_canonical_internal_date_and_timestamp(
        self,
    ) -> None:
        raw_msg = {
            "id": "test-canonical-1",
            "threadId": "thread-1",
            "internalDate": "1710000000000",
            "labelIds": ["INBOX", "UNREAD"],
            "payload": {
                "headers": [
                    {"name": "Subject", "value": "Test Subject"},
                    {"name": "From", "value": "Sender <sender@example.com>"},
                    {
                        "name": "Date",
                        "value": "Sun, 10 Mar 2024 10:00:00 +0000",
                    },
                ],
                "body": {"data": ""},
            },
        }
        normalized = normalize_message(raw_msg)
        self.assertEqual(normalized.message_id, "test-canonical-1")
        self.assertEqual(normalized.internal_date, 1710000000000)
        self.assertTrue(normalized.timestamp.startswith("2024-03-09") or normalized.timestamp.startswith("2024-03-10"))
        self.assertEqual(normalized.labels, ["INBOX", "UNREAD"])
        api_dict = normalized.to_api_dict()
        self.assertEqual(api_dict["internal_date"], 1710000000000)

    def test_rag_source_url_and_labels(self) -> None:
        chunk = EmailChunk(
            chunk_id="c1",
            message_id="msg-12345",
            thread_id="thread-1",
            subject="Important Update",
            sender="alert@service.com",
            recipients=["user@example.com"],
            timestamp="2026-09-05T12:00:00+00:00",
            date="2026-09-05T12:00:00+00:00",
            labels=["INBOX", "IMPORTANT"],
            chunk_index=0,
            total_chunks=1,
            text="Update details",
            internal_date=1000,
        )
        source = RAGSource(
            subject=chunk["subject"],
            sender=chunk["sender"],
            date=chunk["date"],
            message_id=chunk["message_id"],
            thread_id=chunk["thread_id"],
            score=0.95,
            labels=tuple(chunk["labels"]),
            gmail_url=f"https://mail.google.com/mail/u/0/#inbox/{chunk['message_id']}",
        )
        d = source.public_dict()
        self.assertEqual(
            d["gmail_url"], "https://mail.google.com/mail/u/0/#inbox/msg-12345"
        )
        self.assertEqual(d["labels"], ["INBOX", "IMPORTANT"])

    def test_search_messages_live_api_pagination_and_query_parameters(self) -> None:
        class MockGmailMessages:
            def __init__(self):
                self.list_calls = []

            def list(self, **kwargs):
                self.list_calls.append(kwargs)
                token = kwargs.get("pageToken")
                if token is None:
                    return type(
                        "Resp",
                        (),
                        {
                            "execute": lambda s: {
                                "messages": [{"id": "m1"}],
                                "nextPageToken": "page-2",
                            }
                        },
                    )()
                elif token == "page-2":
                    return type(
                        "Resp",
                        (),
                        {"execute": lambda s: {"messages": [{"id": "m2"}]}},
                    )()
                return type("Resp", (), {"execute": lambda s: {"messages": []}})()

            def get(self, userId, id, format):
                return type(
                    "Resp",
                    (),
                    {
                        "execute": lambda s: {
                            "id": id,
                            "threadId": f"t-{id}",
                            "internalDate": "1710000000000",
                            "labelIds": ["INBOX", "UNREAD"],
                            "payload": {
                                "headers": [
                                    {"name": "Subject", "value": f"Subject {id}"},
                                    {"name": "From", "value": "test@example.com"},
                                ],
                                "body": {},
                            },
                        }
                    },
                )()

        class MockGmailService:
            def __init__(self):
                self.api = MockGmailMessages()

            def users(self):
                return type("Users", (), {"messages": lambda s: self.api})()

        service = MockGmailService()
        messages, complete = search_messages(
            query="from:Binance KYC",
            max_messages=2,
            page_size=1,
            include_spam_trash=True,
            label_ids=["UNREAD"],
            service=service,
        )
        self.assertEqual(len(messages), 2)
        self.assertTrue(complete)
        self.assertEqual(messages[0].message_id, "m1")
        self.assertEqual(messages[1].message_id, "m2")
        self.assertEqual(service.api.list_calls[0]["q"], "from:Binance KYC")
        self.assertEqual(service.api.list_calls[0]["includeSpamTrash"], True)
        self.assertEqual(service.api.list_calls[0]["labelIds"], ["UNREAD"])
        self.assertIsNone(service.api.list_calls[0].get("pageToken"))
        self.assertEqual(service.api.list_calls[1]["pageToken"], "page-2")


if __name__ == "__main__":
    unittest.main()
