"""MIME parsing for Gmail message payloads.

Two HTML-to-text conversion paths are maintained:

* ``html_to_text`` — retrieval path.  Preserves link URLs inline as
  ``[url]`` markers so that the text stored in Qdrant retains full URL
  context for semantic search.  **Do not change this output** without
  re-embedding the corpus.

* ``html_to_display_text`` — display path.  Converts HTML to clean,
  human-readable text suitable for rendering in a browser.  Link text is
  kept; raw URLs are stripped unless no display text is available (in
  which case the URL itself is kept as plain text).  This output is never
  stored in the vector database.
"""

from __future__ import annotations

import base64
import re
from html.parser import HTMLParser


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# Matches RFC-2822 / Markdown autolink-style angle-bracket URLs:
#   <https://example.com>
# Also matches the common plain-text email artefact:
#   (<https://example.com>)
_AUTOLINK_RE = re.compile(
    r"\(?\s*<(https?://[^>\s]+)>\s*\)?",
    re.IGNORECASE,
)

# Collapse runs of three or more blank lines to two.
_EXCESS_BLANK_RE = re.compile(r"\n{3,}")


def _collapse_whitespace(text: str) -> str:
    """Normalise line-endings and collapse excessive blank lines."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return _EXCESS_BLANK_RE.sub("\n\n", text).strip()


# ---------------------------------------------------------------------------
# RETRIEVAL HTML parser  (output stored in Qdrant — do not change semantics)
# ---------------------------------------------------------------------------

class _RetrievalHTMLParser(HTMLParser):
    """Extract text + inline URL markers for retrieval/embedding.

    Keeps URLs as ``[url]`` tokens immediately *after* the anchor text so
    that semantic search can match on URL content when relevant.
    """

    _block_tags = {"br", "div", "li", "p", "tr", "h1", "h2", "h3", "h4", "h5", "h6"}
    _skip_tags = {"script", "style", "head"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.fragments: list[str] = []
        self._skip_depth: int = 0
        self._current_href: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._skip_tags:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag in self._block_tags:
            self.fragments.append("\n")
        if tag == "a":
            href = dict(attrs).get("href", "")
            if href and not href.startswith(("#", "mailto:", "tel:", "javascript:")):
                self._current_href = href

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        self.fragments.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in self._skip_tags:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth:
            return
        if tag in self._block_tags:
            self.fragments.append("\n")
        if tag == "a" and self._current_href:
            # Append URL *after* the link text so it remains searchable.
            self.fragments.append(f" [{self._current_href}]")
            self._current_href = None


# ---------------------------------------------------------------------------
# DISPLAY HTML parser  (output shown to the user — never stored in Qdrant)
# ---------------------------------------------------------------------------

class _DisplayHTMLParser(HTMLParser):
    """Convert HTML email to clean, human-readable plain text.

    Rules:
    * Block elements introduce newlines.
    * ``<a>`` links: render the visible link text only.  If the anchor has
      no visible text, fall back to the href URL as plain text.
    * ``<img>`` alt text is preserved as ``[Image: …]`` when meaningful.
    * ``<script>``, ``<style>``, ``<head>`` content is discarded entirely.
    * HTML entities are decoded (``convert_charrefs=True``).
    """

    _block_tags = {"br", "div", "li", "p", "tr", "td", "th",
                   "h1", "h2", "h3", "h4", "h5", "h6",
                   "article", "section", "header", "footer", "main",
                   "blockquote", "pre", "ul", "ol", "table"}
    _skip_tags = {"script", "style", "head", "noscript"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.fragments: list[str] = []
        self._skip_depth: int = 0
        # Stack of (href, anchor_start_index) for nested anchors
        self._anchor_stack: list[tuple[str, int]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._skip_tags:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag in self._block_tags:
            # Ensure we don't double-up on newlines
            if self.fragments and self.fragments[-1] != "\n":
                self.fragments.append("\n")
        elif tag == "a":
            attr_dict = dict(attrs)
            href = attr_dict.get("href", "") or ""
            href = href.strip()
            if href and not href.startswith(("#", "mailto:", "tel:", "javascript:")):
                # Record the href and the position in fragments where anchor text begins
                self._anchor_stack.append((href, len(self.fragments)))
            else:
                self._anchor_stack.append(("", len(self.fragments)))
        elif tag == "img":
            attr_dict = dict(attrs)
            alt = (attr_dict.get("alt") or "").strip()
            if alt:
                self.fragments.append(f"[Image: {alt}]")
        elif tag == "br":
            self.fragments.append("\n")
        elif tag == "hr":
            self.fragments.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        self.fragments.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in self._skip_tags:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth:
            return
        if tag in self._block_tags:
            if self.fragments and self.fragments[-1] != "\n":
                self.fragments.append("\n")
        elif tag == "a" and self._anchor_stack:
            href, start_idx = self._anchor_stack.pop()
            if href:
                # Extract the text that was accumulated inside this anchor
                anchor_text = "".join(self.fragments[start_idx:]).strip()
                # Replace fragments from start_idx onwards with clean link text.
                del self.fragments[start_idx:]
                if anchor_text:
                    # Just use the display text; drop the URL from visible output
                    self.fragments.append(anchor_text)
                else:
                    # No display text — show the URL itself as plain text
                    self.fragments.append(href)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def decode_payload_data(data: str | None) -> str:
    if not data:
        return ""
    try:
        decoded = base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))
    except (ValueError, TypeError):
        return ""
    return decoded.decode("utf-8", errors="replace")


def html_to_text(value: str) -> str:
    """Retrieval path: HTML → text with inline ``[url]`` markers.

    **Do not modify the output format.**  Existing Qdrant vectors were
    built from this representation.
    """
    parser = _RetrievalHTMLParser()
    parser.feed(value)
    parser.close()
    return "".join(parser.fragments)


def html_to_display_text(value: str) -> str:
    """Display path: HTML → clean human-readable text.

    Link text is preserved; raw URLs are stripped from visible output
    (unless an anchor has no visible text, in which case the URL is kept
    as plain text).  Never stored in the vector database.
    """
    parser = _DisplayHTMLParser()
    parser.feed(value)
    parser.close()
    raw = "".join(parser.fragments)
    return _collapse_whitespace(raw)


def clean_autolinks_for_display(text: str) -> str:
    """Remove angle-bracket autolink artefacts from plain-text email bodies.

    Converts patterns like ``<https://example.com>`` and
    ``(<https://example.com>)`` to bare URLs or removes them when they
    appear immediately after their own display text.

    This is the display-only cleaning path for plain-text emails.
    **Do not apply to retrieval text.**
    """

    def _replace(match: re.Match) -> str:
        url = match.group(1)
        full = match.group(0)
        # If wrapped in parens, remove entirely (URL was a parenthetical aside)
        if full.startswith("(") and full.endswith(")"):
            return ""
        # Otherwise keep the bare URL
        return url

    cleaned = _AUTOLINK_RE.sub(_replace, text)
    # After removing autolinks the surrounding text may have orphaned
    # punctuation like trailing commas or double-spaces.
    cleaned = re.sub(r" {2,}", " ", cleaned)
    cleaned = re.sub(r" ([,;.])", r"\1", cleaned)
    return _collapse_whitespace(cleaned)


def extract_body(payload: dict) -> str:
    """Extract the authoritative retrieval body from a Gmail MIME payload.

    Prefers ``text/plain`` parts.  Falls back to HTML parts converted via
    the *retrieval* parser (``html_to_text``).

    **This is the body stored in Qdrant.  Do not change its output.**
    """
    plain_parts: list[str] = []
    html_parts: list[str] = []

    def visit(part: dict) -> None:
        mime_type = part.get("mimeType", "").lower()
        data = decode_payload_data(part.get("body", {}).get("data"))
        if data and mime_type == "text/plain":
            plain_parts.append(data)
        elif data and mime_type == "text/html":
            html_parts.append(data)
        for child in part.get("parts", []) or []:
            visit(child)

    visit(payload)
    if plain_parts:
        return "\n\n".join(plain_parts)
    return "\n\n".join(html_to_text(value) for value in html_parts)


def extract_display_body(payload: dict) -> str:
    """Extract a clean, human-readable body for UI display.

    Prefers ``text/plain`` parts cleaned of autolink artefacts.
    Falls back to HTML parts converted via the *display* parser
    (``html_to_display_text``).

    **Never stored in Qdrant.  Display only.**
    """
    plain_parts: list[str] = []
    html_parts: list[str] = []

    def visit(part: dict) -> None:
        mime_type = part.get("mimeType", "").lower()
        data = decode_payload_data(part.get("body", {}).get("data"))
        if data and mime_type == "text/plain":
            plain_parts.append(data)
        elif data and mime_type == "text/html":
            html_parts.append(data)
        for child in part.get("parts", []) or []:
            visit(child)

    visit(payload)

    if plain_parts:
        raw = "\n\n".join(plain_parts)
        return clean_autolinks_for_display(raw)
    return "\n\n".join(html_to_display_text(value) for value in html_parts)