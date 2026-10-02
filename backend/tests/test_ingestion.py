import base64
import unittest

from backend.app.mime import extract_body
from backend.app.preprocessing import normalize_body


def encoded(value: str) -> str:
    return base64.urlsafe_b64encode(value.encode()).decode().rstrip("=")


class MimeExtractionTests(unittest.TestCase):
    def test_plain_text_message(self) -> None:
        payload = {"mimeType": "text/plain", "body": {"data": encoded("Hello\nworld")}}
        self.assertEqual(extract_body(payload), "Hello\nworld")

    def test_nested_multipart_prefers_plain_text(self) -> None:
        payload = {
            "mimeType": "multipart/mixed",
            "parts": [
                {
                    "mimeType": "multipart/alternative",
                    "parts": [
                        {"mimeType": "text/html", "body": {"data": encoded("HTML")}},
                        {"mimeType": "text/plain", "body": {"data": encoded("Plain text")}},
                    ],
                }
            ],
        }
        self.assertEqual(extract_body(payload), "Plain text")

    def test_html_only_message_becomes_readable_text(self) -> None:
        payload = {
            "mimeType": "text/html",
            "body": {"data": encoded("<p>Hello <b>there</b></p><a href='https://example.com'>Read</a>")},
        }
        self.assertIn("Hello there", extract_body(payload))
        self.assertIn("https://example.com", extract_body(payload))

    def test_empty_body_is_safe(self) -> None:
        self.assertEqual(extract_body({"mimeType": "text/plain", "body": {}}), "")
        self.assertEqual(normalize_body("  one\r\n\r\n\r\n two  "), "one\n\ntwo")


if __name__ == "__main__":
    unittest.main()