import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.rag_service import RAGResult, RAGSource


class ChatAPITests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)
        self.credentials = SimpleNamespace(valid=True)

    def test_unauthenticated_chat_returns_401(self) -> None:
        with patch("backend.app.main.load_credentials", return_value=None):
            response = self.client.post("/api/chat", json={"query": "What is my fee?"})
        self.assertEqual(response.status_code, 401)

    def test_unauthenticated_emails_returns_401(self) -> None:
        with patch("backend.app.main.load_credentials", return_value=None):
            response = self.client.get("/api/emails")
        self.assertEqual(response.status_code, 401)

    def test_unauthenticated_sync_returns_401(self) -> None:
        with patch("backend.app.main.load_credentials", return_value=None):
            response = self.client.post("/api/sync")
        self.assertEqual(response.status_code, 401)

    def test_logout_clears_credentials(self) -> None:
        with patch("backend.app.main.clear_credentials") as clear:
            response = self.client.post("/api/auth/logout")
        self.assertEqual(response.status_code, 200)
        clear.assert_called_once_with()

    def test_supported_chat_returns_answer_and_safe_source(self) -> None:
        result = RAGResult(
            query="What is my fee?", answer="The fee is 105000 rupees.",
            decision="allow", reason="grounded_answer_generated",
            sources=(RAGSource("Fee notice", "university@example.com", "2026-09-05", "message-1", "thread-1", 0.8),),
        )
        with (
            patch("backend.app.main.load_credentials", return_value=self.credentials),
            patch("backend.app.main.generate_gmail_answer", return_value=result) as generate,
        ):
            response = self.client.post(
                "/api/chat",
                json={"query": "What is my fee?", "user_id": "other-user"},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "The fee is 105000 rupees.")
        self.assertEqual(len(response.json()["sources"]), 1)
        self.assertNotIn("body", response.json()["sources"][0])
        generate.assert_called_once_with("personal-gmail", "What is my fee?")

    def test_unrelated_chat_does_not_call_groq(self) -> None:
        result = RAGResult(
            query="What is the capital of France?", answer=None, decision="reject",
            reason="unrelated_to_gmail",
        )
        with (
            patch("backend.app.main.load_credentials", return_value=self.credentials),
            patch("backend.app.main.generate_gmail_answer", return_value=result),
        ):
            response = self.client.post("/api/chat", json={"query": "What is the capital of France?"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["answer"],
            "I can only answer questions based on information available in your Gmail.",
        )

    def test_insufficient_evidence_chat_returns_safe_message(self) -> None:
        result = RAGResult(
            query="What is my passport number?", answer=None, decision="reject",
            reason="insufficient_gmail_evidence",
        )
        with (
            patch("backend.app.main.load_credentials", return_value=self.credentials),
            patch("backend.app.main.generate_gmail_answer", return_value=result),
        ):
            response = self.client.post("/api/chat", json={"query": "What is my passport number?"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["answer"],
            "I couldn't find enough information in your Gmail to answer that.",
        )

    def test_backend_failure_hides_internal_error(self) -> None:
        with (
            patch("backend.app.main.load_credentials", return_value=self.credentials),
            patch("backend.app.main.generate_gmail_answer", side_effect=RuntimeError("secret details")),
        ):
            response = self.client.post("/api/chat", json={"query": "What is my fee?"})
        self.assertEqual(response.status_code, 502)
        self.assertNotIn("secret details", response.text)

    def test_sync_uses_authenticated_user_and_returns_stats(self) -> None:
        class FakeSync:
            def sync(self, user_id):
                self.user_id = user_id
                return type("Stats", (), {"public_dict": lambda self: {"messages_new": 2}})()

        sync = FakeSync()
        with (
            patch("backend.app.main.load_credentials", return_value=self.credentials),
            patch("backend.app.main.GmailSyncService", return_value=sync),
        ):
            response = self.client.post("/api/sync", json={"user_id": "other-user"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"messages_new": 2})
        self.assertEqual(sync.user_id, "personal-gmail")


if __name__ == "__main__":
    unittest.main()