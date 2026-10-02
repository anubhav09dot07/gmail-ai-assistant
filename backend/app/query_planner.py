from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from .query_normalizer import normalize_query

_TOKEN_PATTERN = re.compile(r"[a-z0-9']+")
_GMAIL_TERMS = {
    "email",
    "emails",
    "gmail",
    "message",
    "messages",
    "thread",
    "inbox",
    "sender",
    "subject",
}
_PERSONAL_REFERENCES = {"i", "my", "me", "our", "we", "they", "them", "latest"}
_EMAIL_ACTIONS = {
    "said",
    "say",
    "sent",
    "send",
    "mentioned",
    "decide",
    "decided",
    "discussed",
    "wrote",
    "received",
    "forwarded",
    "log",
    "logged",
    "logging",
}
_SEARCH_SIGNALS = {
    "from",
    "latest",
    "last",
    "recent",
    "newest",
    "received",
    "sent",
    "yesterday",
    "today",
    "week",
    "month",
    "unread",
    "spam",
    "trash",
    "promotions",
}
_OBVIOUS_NON_GMAIL_PATTERNS = (
    re.compile(r"\bwhat is the capital of\b"),
    re.compile(r"\bwho is the president of\b"),
    re.compile(
        r"\b(?:explain|how does|how do|how is)\s+(?:quantum mechanics|relativity|photosynthesis)\b"
    ),
    re.compile(r"\bwrite(?: me)?\b.*\b(?:python|program|code|script)\b"),
    re.compile(r"\b(?:today'?s|tomorrow'?s|recent|latest) weather\b"),
    re.compile(r"\bwho won\b.*\b(?:match|game|cricket|football)\b"),
    re.compile(r"\btell me a joke\b"),
)

_QUERY_STRUCTURE_TERMS = {
    "a",
    "an",
    "and",
    "any",
    "did",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "me",
    "my",
    "of",
    "on",
    "show",
    "the",
    "there",
    "to",
    "was",
    "what",
    "when",
    "which",
    "who",
    "with",
    "you",
    "your",
    "last",
    "latest",
    "most",
    "recent",
    "recently",
    "newest",
    "yesterday",
    "today",
    "this",
    "week",
    "mail",
    "mails",
    "email",
    "emails",
    "message",
    "messages",
    "thread",
    "threads",
    "sign",
    "account",
    "received",
    "receive",
    "sent",
    "send",
    "said",
    "say",
    "mentioned",
    "decide",
    "decided",
    "wrote",
    "forwarded",
    "work",
    "much",
    "pay",
    "paid",
    "that",
    "about",
    "someone",
    "log",
    "logged",
    "logging",
    "signing",
    "into",
    "new",
    "detected",
}

_STOP_WORDS_FOR_TOPIC = {
    "a",
    "an",
    "the",
    "my",
    "our",
    "your",
    "me",
    "i",
    "we",
    "they",
    "them",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "by",
    "from",
    "about",
    "regarding",
    "mentioning",
    "email",
    "emails",
    "message",
    "messages",
    "mail",
    "mails",
    "show",
    "get",
    "find",
    "give",
    "what",
    "who",
    "which",
    "when",
    "where",
    "how",
    "did",
    "does",
    "is",
    "are",
    "was",
    "were",
    "be",
    "any",
    "all",
    "latest",
    "last",
    "newest",
    "most",
    "recent",
    "recently",
    "unread",
    "spam",
    "trash",
    "bin",
    "junk",
    "related",
    "relation",
}

_ENTITY_IGNORE_WORDS = {
    "a", "an", "and", "any", "are", "did", "email", "emails", "find",
    "from", "how", "i", "in", "is", "mail", "message", "messages", "my",
    "recent", "related", "show", "the", "to", "what", "when", "where", "who",
}


@dataclass(frozen=True)
class SearchPlan:
    raw_query: str
    is_gmail_related: bool
    # Structured facets
    sender: str | None = None
    recipient: str | None = None
    subject: str | None = None
    entities: tuple[str, ...] = ()
    has_attachment: bool = False
    labels: tuple[str, ...] = ()
    gmail_query: str = ""
    include_spam_trash: bool = False
    # Temporal & ordering facets
    sort: str = "relevance"  # "newest" | "relevance"
    limit: int = 5
    date_range: str | None = None
    internal_date_after: int | None = None
    internal_date_before: int | None = None
    # Semantic facets
    semantic_query: str = ""
    # Execution flags
    use_structured_search: bool = False
    use_semantic_search: bool = False
    reason: str = ""

    @property
    def retrieval_mode(self) -> str:
        if self.use_structured_search and self.use_semantic_search:
            return "hybrid"
        if self.use_structured_search:
            return "structured"
        return "semantic"


def _check_relevance(raw_query: str) -> tuple[bool, str]:
    normalized = normalize_query(raw_query)
    tokens = set(_TOKEN_PATTERN.findall(normalized))
    if any(pattern.search(normalized) for pattern in _OBVIOUS_NON_GMAIL_PATTERNS):
        return False, "unrelated_to_gmail"

    sign_in_signal = bool(re.search(r"\bsign[\s-]?in\b", normalized))
    explicit_gmail_signal = bool(tokens & _GMAIL_TERMS) or sign_in_signal
    personal_signal = bool(tokens & _PERSONAL_REFERENCES)
    email_action_signal = bool(tokens & _EMAIL_ACTIONS)
    search_signal = bool(tokens & _SEARCH_SIGNALS)
    content_signal = len(tokens - _QUERY_STRUCTURE_TERMS) >= 1
    personal_question = personal_signal and bool(
        tokens & {"what", "who", "which", "when", "where", "how"}
    )
    relational_question = bool(
        re.search(r"\b(?:what|who|which|when|where|did|does)\b", normalized)
        and re.search(
            r"\b(?:said|say|sent|mentioned|decide|decided|deadline|amount)\b",
            normalized,
        )
    )

    topic_phrase = len(tokens - _QUERY_STRUCTURE_TERMS) >= 2
    if (
        explicit_gmail_signal
        or topic_phrase
        or (search_signal and content_signal)
        or (
            personal_signal
            and (
                email_action_signal
                or relational_question
                or personal_question
                or content_signal
            )
        )
    ):
        return True, "plausibly_about_gmail"
    if re.search(r"\bwhat did (?:they|we) finally decide\b", normalized):
        return True, "plausibly_about_gmail"
    return False, "unrelated_to_gmail"


def _gmail_date_boundary(value: str, *, end_of_day: bool = False) -> int | None:
    """Convert a Gmail YYYY/MM/DD search boundary to UTC epoch milliseconds."""
    try:
        day = datetime.strptime(value.replace("-", "/"), "%Y/%m/%d").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        return None
    if end_of_day:
        day += timedelta(days=1)
    return int(day.timestamp() * 1000)


def _native_operator(raw_query: str, name: str) -> str | None:
    """Read a Gmail operator without interpreting its value as natural-language text."""
    match = re.search(
        rf"(?<!\S){re.escape(name)}:(\"[^\"]+\"|\S+)", raw_query, re.IGNORECASE
    )
    if not match:
        return None
    return match.group(1).strip('"')

def _explicit_entities(raw_query: str) -> tuple[str, ...]:
    """Extract explicit names/acronyms without treating sentence-initial words as entities."""
    entities: list[str] = []

    for match in re.finditer(
        r"\b(?:[A-Z]{3,}|[A-Z][a-z]{2,})\b",
        raw_query,
    ):
        value = match.group(0)

        if value.lower() in _ENTITY_IGNORE_WORDS:
            continue

        # Ignore normal Title Case words when they merely start the sentence.
        # Keep acronyms such as SBI.
        if match.start() == 0 and not value.isupper():
            continue

        if value not in entities:
            entities.append(value)

    return tuple(entities)


def plan_search(raw_query: str) -> SearchPlan:
    """Analyze query and produce a multi-faceted SearchPlan with orthogonal facets."""
    clean_query = raw_query.strip() if isinstance(raw_query, str) else ""
    if not clean_query:
        return SearchPlan(
            raw_query=clean_query,
            is_gmail_related=False,
            reason="empty_query",
        )

    is_related, relevance_reason = _check_relevance(clean_query)
    if not is_related:
        return SearchPlan(
            raw_query=clean_query,
            is_gmail_related=False,
            reason=relevance_reason,
        )

    normalized = normalize_query(clean_query)
    tokens = set(_TOKEN_PATTERN.findall(normalized))
    # Treat names as hard content constraints when the wording explicitly asks
    # for messages related/about/regarding that name.  This avoids mistaking
    # every capitalized word in an ordinary question for a mailbox filter.
    entities = (
        _explicit_entities(clean_query)
        if re.search(r"\b(?:related|about|regarding|mentioning)\b", normalized)
        else ()
    )

    # 1. Temporal / Ordering extraction
    sort = "relevance"
    limit = 5
    if any(term in tokens for term in ("latest", "newest", "recent", "recently")) or re.search(
        r"\b(?:most recent|last email)\b", normalized
    ):
        sort = "newest"

    limit_match = re.search(r"\b(?:last|latest|recent)\s+(\d+)\s+emails?\b", normalized)
    if limit_match:
        try:
            limit = max(1, int(limit_match.group(1)))
            sort = "newest"
        except ValueError:
            pass
    elif re.search(r"\b(?:latest|most recent|last)\s+email\b", normalized):
        limit = 1
        sort = "newest"

    # 2. Envelope and subject extraction.  Explicit Gmail syntax is retained
    # verbatim; natural language is converted into the equivalent operator.
    sender: str | None = _native_operator(clean_query, "from")
    recipient: str | None = _native_operator(clean_query, "to")
    subject: str | None = _native_operator(clean_query, "subject")
    gmail_query_parts: list[str] = []

    if not sender:
        # Check natural language "from (my |the )?([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}|[a-zA-Z0-9]+)"
        nl_from_match = re.search(
            r"\bfrom\s+(?:my\s+|the\s+)?([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}|[a-zA-Z0-9]+)\b",
            clean_query,
            re.IGNORECASE,
        )
        if nl_from_match:
            candidate_sender = nl_from_match.group(1).strip()
            # Ignore temporal/filler words following from (e.g., "from yesterday")
            if candidate_sender.lower() not in {"yesterday", "today", "me", "here", "last", "any"}:
                sender = candidate_sender

    if sender:
        gmail_query_parts.append(f"from:{sender}")

    if not recipient:
        to_match = re.search(
            r"\bto\s+([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b",
            clean_query,
            re.IGNORECASE,
        )
        if to_match:
            recipient = to_match.group(1).strip()
    if recipient:
        gmail_query_parts.append(f"to:{recipient}")

    if not subject:
        subject_match = re.search(
            r"\bsubject\s+(?:is\s+)?[\"']?([^\"'?.,]+)", clean_query,
            re.IGNORECASE,
        )
        if subject_match:
            subject = subject_match.group(1).strip()
    if subject:
        quoted_subject = f'"{subject}"' if " " in subject else subject
        gmail_query_parts.append(f"subject:{quoted_subject}")

    # “Related to <name>” is a content/sender/subject constraint, not a
    # sender-only assumption. Gmail's bare-term search covers all of those.
    gmail_query_parts.extend(entities)

    # 3. Label extraction
    labels_list: list[str] = []
    include_spam_trash = False

    if re.search(r"\b(?:spam|junk|in:spam)\b", normalized):
        labels_list.append("SPAM")
        include_spam_trash = True
        gmail_query_parts.append("in:spam")
    if re.search(r"\b(?:trash|bin|in:trash)\b", normalized):
        labels_list.append("TRASH")
        include_spam_trash = True
        gmail_query_parts.append("in:trash")
    if re.search(r"\b(?:unread|not read|is:unread)\b", normalized):
        labels_list.append("UNREAD")
        gmail_query_parts.append("is:unread")
    if re.search(r"\b(?:starred|is:starred)\b", normalized):
        labels_list.append("STARRED")
        gmail_query_parts.append("is:starred")
    if re.search(r"\b(?:promotions?|promotional|category:promotions?)\b", normalized):
        labels_list.append("CATEGORY_PROMOTIONS")
        gmail_query_parts.append("category:promotions")

    # Gmail "Purchases" is a category search operator, not a labelIds filter.
    # Therefore use category:purchases without adding CATEGORY_PURCHASES
    # to labels_list.
    if re.search(r"\b(?:purchases?|purchase|category:purchases?)\b", normalized):
        gmail_query_parts.append("category:purchases")

    if re.search(r"\b(?:social|category:social)\b", normalized):
        labels_list.append("CATEGORY_SOCIAL")
        gmail_query_parts.append("category:social")
    if re.search(r"\b(?:updates?|category:updates?)\b", normalized):
        labels_list.append("CATEGORY_UPDATES")
        gmail_query_parts.append("category:updates")
    if re.search(r"\b(?:forums?|category:forums?)\b", normalized):
        labels_list.append("CATEGORY_FORUMS")
        gmail_query_parts.append("category:forums")

    has_attachment = bool(
        re.search(r"\b(?:has:attachment|with (?:an )?attachment|attachments?)\b", normalized)
    )
    if has_attachment:
        gmail_query_parts.append("has:attachment")

    # Native date filters keep Gmail as the authoritative structured search
    # engine, while the matching epoch boundaries let Qdrant apply the same
    # constraints to its already-indexed message metadata.
    date_range: str | None = None
    internal_date_after: int | None = None
    internal_date_before: int | None = None
    after = _native_operator(clean_query, "after")
    before = _native_operator(clean_query, "before")
    newer_than = _native_operator(clean_query, "newer_than")
    older_than = _native_operator(clean_query, "older_than")
    if after:
        gmail_query_parts.append(f"after:{after}")
        internal_date_after = _gmail_date_boundary(after)
        date_range = f"after:{after}"
    if before:
        gmail_query_parts.append(f"before:{before}")
        internal_date_before = _gmail_date_boundary(before)
        date_range = f"{date_range or ''} before:{before}".strip()
    if newer_than:
        gmail_query_parts.append(f"newer_than:{newer_than}")
        date_range = f"newer_than:{newer_than}"
        relative = re.fullmatch(r"(\d+)([dmy])", newer_than.lower())
        if relative:
            amount, unit = int(relative.group(1)), relative.group(2)
            days = amount * {"d": 1, "m": 30, "y": 365}[unit]
            internal_date_after = int(
                (datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000
            )
    if older_than:
        gmail_query_parts.append(f"older_than:{older_than}")
        date_range = f"older_than:{older_than}"
        relative = re.fullmatch(r"(\d+)([dmy])", older_than.lower())
        if relative:
            amount, unit = int(relative.group(1)), relative.group(2)
            days = amount * {"d": 1, "m": 30, "y": 365}[unit]
            internal_date_before = int(
                (datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000
            )
    if not any((after, before, newer_than, older_than)):
        if re.search(r"\bthis week\b", normalized):
            gmail_query_parts.append("newer_than:7d")
            date_range = "newer_than:7d"
            internal_date_after = int(
                (datetime.now(timezone.utc) - timedelta(days=7)).timestamp() * 1000
            )
        elif re.search(r"\btoday\b", normalized):
            today_date = datetime.now(timezone.utc).date()
            today = today_date.isoformat().replace("-", "/")
            gmail_query_parts.append(f"after:{today}")
            internal_date_after = _gmail_date_boundary(today)
            date_range = f"after:{today}"

        elif re.search(r"\byesterday\b", normalized):
            today_date = datetime.now(timezone.utc).date()
            yesterday_date = today_date - timedelta(days=1)

            yesterday = yesterday_date.isoformat().replace("-", "/")
            today = today_date.isoformat().replace("-", "/")

            gmail_query_parts.extend([
                f"after:{yesterday}",
                f"before:{today}",
            ])
            internal_date_after = _gmail_date_boundary(yesterday)
            internal_date_before = _gmail_date_boundary(today)
            date_range = f"after:{yesterday} before:{today}"

        elif re.search(r"\bthis month\b", normalized):
            today_date = datetime.now(timezone.utc).date()
            month_start = today_date.replace(day=1)

            if month_start.month == 12:
                next_month = month_start.replace(
                    year=month_start.year + 1,
                    month=1,
                    day=1,
                )
            else:
                next_month = month_start.replace(
                    month=month_start.month + 1,
                    day=1,
                )

            start = month_start.isoformat().replace("-", "/")
            end = next_month.isoformat().replace("-", "/")

            gmail_query_parts.extend([
                f"after:{start}",
                f"before:{end}",
            ])
            internal_date_after = _gmail_date_boundary(start)
            internal_date_before = _gmail_date_boundary(end)
            date_range = f"after:{start} before:{end}"

    # 4. Semantic Query extraction
    # Strip structured markers and wrappers from the raw query
    semantic_text = clean_query

    # Extract explicit topic clause like "about ...", "regarding ...", "mentioning ..."
    topic_match = re.search(r"\b(?:about|regarding|mentioning)\s+(.+)$", clean_query, re.IGNORECASE)
    if topic_match:
        semantic_text = topic_match.group(1).strip()
    else:
        # Remove sender phrase
        if sender:
            semantic_text = re.sub(
                rf"\bfrom\s+(?:my\s+|the\s+)?{re.escape(sender)}\b",
                "",
                semantic_text,
                flags=re.IGNORECASE,
            )
            semantic_text = re.sub(
                rf"\bfrom:{re.escape(sender)}\b",
                "",
                semantic_text,
                flags=re.IGNORECASE,
            )
        if recipient:
            semantic_text = re.sub(
                rf"\bto\s+{re.escape(recipient)}\b|\bto:{re.escape(recipient)}\b",
                "",
                semantic_text,
                flags=re.IGNORECASE,
            )
        if subject:
            semantic_text = re.sub(
                rf"\bsubject(?::|\s+(?:is\s+)?)\"?{re.escape(subject)}\"?",
                "",
                semantic_text,
                flags=re.IGNORECASE,
            )
        # Remove email envelope keywords
        semantic_text = re.sub(
            r"\b(?:latest|newest|recent|recently|most recent|last)\b",
            "",
            semantic_text,
            flags=re.IGNORECASE,
        )
        semantic_text = re.sub(
            r"\b(?:email|emails|message|messages|mail|mails)\b",
            "",
            semantic_text,
            flags=re.IGNORECASE,
        )
        semantic_text = re.sub(
            r"\b(?:in\s+spam|in\s+trash|unread|starred|promotions?|updates?|social|forums?|has:attachment|attachments?|today|this\s+week)\b",
            "",
            semantic_text,
            flags=re.IGNORECASE,
        )
        semantic_text = re.sub(
            r"\b(?:after|before|newer_than|older_than):\S+\b",
            "",
            semantic_text,
            flags=re.IGNORECASE,
        )

    # Clean residual whitespace and punctuation
    semantic_text = re.sub(r"^[,\s:?.-]+|[,\s:?.-]+$", "", semantic_text).strip()
    # Check if there are meaningful non-stopword tokens remaining
    topic_tokens = [
        t for t in _TOKEN_PATTERN.findall(semantic_text.lower())
        if t not in _STOP_WORDS_FOR_TOPIC and len(t) > 1
    ]

    semantic_query = " ".join(topic_tokens) if topic_tokens else ""

    full_gmail_query = " ".join(gmail_query_parts).strip()

    # Determine execution flags
    use_structured_search = bool(
        gmail_query_parts
        or sender
        or recipient
        or subject
        or entities
        or labels_list
        or has_attachment
        or date_range
        or sort == "newest"
        or include_spam_trash
    )
    use_semantic_search = bool(semantic_query)

    # If neither flag is set, default to semantic search on the original query
    if not use_structured_search and not use_semantic_search:
        use_semantic_search = True
        semantic_query = clean_query

    return SearchPlan(
        raw_query=clean_query,
        is_gmail_related=True,
        sender=sender,
        recipient=recipient,
        subject=subject,
        entities=entities,
        has_attachment=has_attachment,
        labels=tuple(labels_list),
        gmail_query=full_gmail_query,
        include_spam_trash=include_spam_trash,
        sort=sort,
        limit=limit,
        date_range=date_range,
        internal_date_after=internal_date_after,
        internal_date_before=internal_date_before,
        semantic_query=semantic_query,
        use_structured_search=use_structured_search,
        use_semantic_search=use_semantic_search,
        reason="plausibly_about_gmail",
    )
