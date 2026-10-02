"""Text preprocessing utilities for Gmail message bodies.

Two body cleaning functions are provided:

* ``normalize_body`` — retrieval/storage path.  Collapses excess whitespace
  while preserving all content.  **Applied before chunking/embedding.**

* ``normalize_display_body`` — display path.  Applies ``normalize_body``
  then strips any residual autolink artefacts (``<url>``) left over from
  plain-text emails that were not HTML-parsed.  **Never stored in Qdrant.**
"""

import re

from .mime import clean_autolinks_for_display


def normalize_body(value: str) -> str:
    """Collapse excess whitespace for retrieval storage.

    Preserves all meaningful content (URLs, entities, punctuation).
    """
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in value.split("\n")]
    cleaned_lines: list[str] = []
    previous_blank = False
    for line in lines:
        if not line:
            if not previous_blank and cleaned_lines:
                cleaned_lines.append("")
            previous_blank = True
            continue
        cleaned_lines.append(line)
        previous_blank = False
    return "\n".join(cleaned_lines).strip()


def normalize_display_body(value: str) -> str:
    """Normalize and clean a plain-text body for human-readable display.

    Applies ``normalize_body`` then removes autolink angle-bracket
    artefacts (``<https://...>`` / ``(<https://...>)``) that appear in
    RFC-2822 plain-text emails.

    **Display only — never stored in Qdrant.**
    """
    normalized = normalize_body(value)
    return clean_autolinks_for_display(normalized)