"""Tests for the display-layer HTML cleaning and autolink stripping.

Covers:
- HTML link rendering (link text kept, raw URL hidden)
- Autolink angle-bracket artefacts stripped from plain-text
- Parenthesised autolinks removed
- Plain-text emails preserved as-is
- HTML entities decoded
- Script/style injection neutralised
- Img alt-text preserved or dropped appropriately
- Block-element → newline conversion
- Retrieval body unchanged (backward-compat)
"""

import base64
import unittest

from backend.app.mime import (
    clean_autolinks_for_display,
    extract_body,
    extract_display_body,
    html_to_display_text,
    html_to_text,
)
from backend.app.preprocessing import normalize_display_body


def enc(value: str) -> str:
    return base64.urlsafe_b64encode(value.encode()).decode().rstrip("=")


# ---------------------------------------------------------------------------
# html_to_display_text tests
# ---------------------------------------------------------------------------

class HtmlToDisplayTextTests(unittest.TestCase):

    def test_simple_anchor_shows_link_text_only(self) -> None:
        html = '<p>This is <a href="https://example.com">Terms of Service</a>.</p>'
        result = html_to_display_text(html)
        self.assertIn("Terms of Service", result)
        self.assertNotIn("https://example.com", result)
        self.assertNotIn("<a", result)

    def test_anchor_with_no_text_falls_back_to_url(self) -> None:
        html = '<a href="https://example.com"></a>'
        result = html_to_display_text(html)
        self.assertIn("https://example.com", result)

    def test_multiple_anchors_all_text_shown(self) -> None:
        html = (
            '<p>Read our <a href="https://a.com">Privacy Policy</a> and '
            '<a href="https://b.com">Terms of Service</a>.</p>'
        )
        result = html_to_display_text(html)
        self.assertIn("Privacy Policy", result)
        self.assertIn("Terms of Service", result)
        self.assertNotIn("https://a.com", result)
        self.assertNotIn("https://b.com", result)

    def test_script_tag_content_removed(self) -> None:
        html = '<script>alert("xss")</script><p>Hello</p>'
        result = html_to_display_text(html)
        self.assertIn("Hello", result)
        self.assertNotIn("alert", result)
        self.assertNotIn("xss", result)
        self.assertNotIn("<script", result)

    def test_style_tag_content_removed(self) -> None:
        html = "<style>body { color: red }</style><p>Content</p>"
        result = html_to_display_text(html)
        self.assertIn("Content", result)
        self.assertNotIn("color", result)

    def test_html_entities_decoded(self) -> None:
        html = "<p>Terms &amp; Conditions</p>"
        result = html_to_display_text(html)
        self.assertIn("Terms & Conditions", result)
        self.assertNotIn("&amp;", result)

    def test_nbsp_becomes_space(self) -> None:
        html = "<p>Hello&nbsp;World</p>"
        result = html_to_display_text(html)
        self.assertIn("Hello", result)
        self.assertIn("World", result)

    def test_block_elements_produce_newlines(self) -> None:
        html = "<div>First</div><div>Second</div>"
        result = html_to_display_text(html)
        self.assertIn("First", result)
        self.assertIn("Second", result)
        # Should have a newline separating them
        self.assertIn("\n", result)

    def test_img_with_alt_text_preserved(self) -> None:
        html = '<img alt="Your boarding pass">'
        result = html_to_display_text(html)
        self.assertIn("Your boarding pass", result)
        self.assertIn("[Image:", result)

    def test_img_with_empty_alt_not_shown(self) -> None:
        html = '<img alt=""><p>Hello</p>'
        result = html_to_display_text(html)
        self.assertNotIn("[Image:", result)
        self.assertIn("Hello", result)

    def test_mailto_link_not_treated_as_href(self) -> None:
        html = '<a href="mailto:user@example.com">Contact us</a>'
        result = html_to_display_text(html)
        self.assertIn("Contact us", result)
        # mailto: is skipped — URL should not appear
        self.assertNotIn("mailto:", result)

    def test_hash_link_not_treated_as_href(self) -> None:
        html = '<a href="#section">Jump</a>'
        result = html_to_display_text(html)
        self.assertIn("Jump", result)
        self.assertNotIn("#section", result)

    def test_no_raw_html_tags_in_output(self) -> None:
        html = "<p><b>Bold</b> and <i>italic</i></p>"
        result = html_to_display_text(html)
        self.assertNotIn("<p>", result)
        self.assertNotIn("<b>", result)
        self.assertNotIn("<i>", result)
        self.assertIn("Bold", result)
        self.assertIn("italic", result)

    def test_youtube_terms_of_service_example(self) -> None:
        """Simulate the actual YouTube ToS email structure."""
        html = (
            "<p>This email is a quarterly reminder that your use of YouTube "
            'is subject to the <a href="https://c.gle/AAAA">Terms of Service</a>, '
            '<a href="https://c.gle/BBBB">Community Guidelines</a> and '
            '<a href="https://c.gle/CCCC">Privacy Policy</a>.</p>'
        )
        result = html_to_display_text(html)
        self.assertIn("Terms of Service", result)
        self.assertIn("Community Guidelines", result)
        self.assertIn("Privacy Policy", result)
        # None of the tracking URLs should appear
        self.assertNotIn("c.gle", result)
        self.assertNotIn("https://", result)


# ---------------------------------------------------------------------------
# clean_autolinks_for_display tests (plain-text email artefacts)
# ---------------------------------------------------------------------------

class CleanAutolinksTests(unittest.TestCase):

    def test_angle_bracket_autolink_stripped_to_bare_url(self) -> None:
        text = "Visit <https://example.com>"
        result = clean_autolinks_for_display(text)
        self.assertNotIn("<https://", result)
        self.assertNotIn(">", result.split("https://example.com")[0])
        self.assertIn("https://example.com", result)

    def test_parenthesised_autolink_removed(self) -> None:
        text = "Community Guidelines (<https://example.com>)"
        result = clean_autolinks_for_display(text)
        self.assertNotIn("<https://", result)
        self.assertIn("Community Guidelines", result)

    def test_plain_text_no_url_unchanged(self) -> None:
        text = "Hello Anubhav,\n\nYour refund is scheduled for September 28.\n\nThanks"
        result = clean_autolinks_for_display(text)
        self.assertIn("Hello Anubhav", result)
        self.assertIn("September 28", result)
        self.assertIn("Thanks", result)

    def test_bare_url_preserved(self) -> None:
        text = "Track it here:\nhttps://example.com/order/123"
        result = clean_autolinks_for_display(text)
        self.assertIn("https://example.com/order/123", result)

    def test_multiple_autolinks_all_cleaned(self) -> None:
        text = (
            "Terms of Service\n"
            "<https://c.gle/AAAA>,\n"
            "Community Guidelines\n"
            "(<https://c.gle/BBBB>)"
        )
        result = clean_autolinks_for_display(text)
        self.assertNotIn("<https://", result)
        self.assertIn("Terms of Service", result)
        self.assertIn("Community Guidelines", result)


# ---------------------------------------------------------------------------
# extract_display_body tests (end-to-end through MIME parser)
# ---------------------------------------------------------------------------

class ExtractDisplayBodyTests(unittest.TestCase):

    def test_html_email_link_text_only(self) -> None:
        html = '<p>Click <a href="https://example.com">here</a>.</p>'
        payload = {"mimeType": "text/html", "body": {"data": enc(html)}}
        result = extract_display_body(payload)
        self.assertIn("here", result)
        self.assertNotIn("https://example.com", result)

    def test_plain_text_email_autolink_stripped(self) -> None:
        plain = "Terms of Service\n<https://c.gle/XXXX>"
        payload = {"mimeType": "text/plain", "body": {"data": enc(plain)}}
        result = extract_display_body(payload)
        self.assertIn("Terms of Service", result)
        self.assertNotIn("<https://", result)

    def test_retrieval_body_unchanged_when_display_diverges(self) -> None:
        html = '<p><a href="https://example.com">Terms of Service</a></p>'
        payload = {"mimeType": "text/html", "body": {"data": enc(html)}}
        retrieval = extract_body(payload)
        display = extract_display_body(payload)
        # Retrieval body must still contain the URL marker
        self.assertIn("https://example.com", retrieval)
        # Display body must not expose the URL
        self.assertNotIn("https://example.com", display)

    def test_prefers_plain_text_over_html(self) -> None:
        payload = {
            "mimeType": "multipart/alternative",
            "parts": [
                {"mimeType": "text/html", "body": {"data": enc("<p>HTML version</p>")}},
                {"mimeType": "text/plain", "body": {"data": enc("Plain version")}},
            ],
        }
        result = extract_display_body(payload)
        self.assertIn("Plain version", result)
        self.assertNotIn("HTML version", result)

    def test_script_injection_neutralised(self) -> None:
        html = '<script>alert("pwned")</script><p>Hello</p>'
        payload = {"mimeType": "text/html", "body": {"data": enc(html)}}
        result = extract_display_body(payload)
        self.assertIn("Hello", result)
        self.assertNotIn("alert", result)
        self.assertNotIn("pwned", result)


# ---------------------------------------------------------------------------
# Retrieval body backward-compatibility tests
# ---------------------------------------------------------------------------

class RetrievalBodyBackwardCompatTests(unittest.TestCase):
    """Ensure html_to_text (retrieval path) output is unchanged."""

    def test_retrieval_includes_url_inline(self) -> None:
        html = '<a href="https://example.com">Click</a>'
        result = html_to_text(html)
        self.assertIn("Click", result)
        self.assertIn("https://example.com", result)

    def test_retrieval_ignores_script_content(self) -> None:
        html = '<script>alert("x")</script><p>Body</p>'
        result = html_to_text(html)
        self.assertIn("Body", result)
        self.assertNotIn("alert", result)

    def test_existing_test_html_only_message_still_passes(self) -> None:
        """Regression: existing test in test_ingestion.py must still pass."""
        payload = {
            "mimeType": "text/html",
            "body": {"data": enc("<p>Hello <b>there</b></p><a href='https://example.com'>Read</a>")},
        }
        result = extract_body(payload)
        self.assertIn("Hello there", result)
        self.assertIn("https://example.com", result)


# ---------------------------------------------------------------------------
# normalize_display_body tests
# ---------------------------------------------------------------------------

class NormalizeDisplayBodyTests(unittest.TestCase):

    def test_normalizes_and_strips_autolinks(self) -> None:
        raw = "Hi,\r\n\r\nCheck here: <https://example.com>"
        result = normalize_display_body(raw)
        self.assertNotIn("<https://", result)
        self.assertIn("Check here:", result)
        self.assertIn("https://example.com", result)

    def test_collapses_excess_blank_lines(self) -> None:
        raw = "A\n\n\n\n\nB"
        result = normalize_display_body(raw)
        self.assertEqual(result.count("\n\n"), 1)

    def test_html_entities_not_decoded_in_plain_text_path(self) -> None:
        """Plain-text emails: &amp; stays as &amp; since Python's html.parser
        is not involved here (no HTML parsing for plain-text)."""
        raw = "Terms &amp; Conditions"
        result = normalize_display_body(raw)
        # We don't run HTML entity decoding on plain text — it stays literal
        self.assertIn("&amp;", result)


if __name__ == "__main__":
    unittest.main()
