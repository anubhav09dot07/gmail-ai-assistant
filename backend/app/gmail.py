from datetime import datetime, timezone
from email.utils import getaddresses, parsedate_to_datetime

from googleapiclient.discovery import build

from . import config
from .mime import extract_body, extract_display_body
from .models import NormalizedGmailMessage
from .oauth import load_credentials
from .preprocessing import normalize_body, normalize_display_body


def _header(headers: list[dict], name: str) -> str:
    target = name.lower()
    return next(
        (
            item.get("value", "")
            for item in headers
            if item.get("name", "").lower() == target
        ),
        "",
    )


def _format_date(value: str) -> str:
    if not value:
        return ""
    try:
        return parsedate_to_datetime(value).isoformat()
    except (TypeError, ValueError):
        return value


def _parse_internal_date(value) -> int:
    if not value:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _format_internal_date(internal_date_ms: int) -> str:
    if not internal_date_ms:
        return ""
    try:
        dt = datetime.fromtimestamp(internal_date_ms / 1000.0, tz=timezone.utc)
        return dt.isoformat()
    except (OverflowError, OSError, ValueError):
        return ""


def _addresses(headers: list[dict], name: str) -> list[str]:
    value = _header(headers, name)
    return [address for _, address in getaddresses([value]) if address]


def normalize_message(message: dict) -> NormalizedGmailMessage:
    payload = message.get("payload", {})
    headers = payload.get("headers", [])
    raw_date_header = _header(headers, "Date")
    parsed_date_header = _format_date(raw_date_header)

    internal_date_ms = _parse_internal_date(message.get("internalDate"))
    if internal_date_ms > 0:
        canonical_timestamp = _format_internal_date(internal_date_ms)
    else:
        canonical_timestamp = parsed_date_header
        if parsed_date_header:
            try:
                dt = datetime.fromisoformat(parsed_date_header)
                internal_date_ms = int(dt.timestamp() * 1000)
            except Exception:
                pass

    # Retrieval body — used by chunking & Qdrant; must not change.
    retrieval_body = normalize_body(extract_body(payload))
    retrieval_snippet = normalize_body(message.get("snippet", ""))

    # Display body — human-readable, never stored in Qdrant.
    display_body = normalize_display_body(extract_display_body(payload))
    display_snippet = normalize_display_body(message.get("snippet", ""))

    return NormalizedGmailMessage(
        message_id=message.get("id", ""),
        thread_id=message.get("threadId", ""),
        subject=_header(headers, "Subject"),
        sender=_header(headers, "From"),
        recipients=_addresses(headers, "To")
        + _addresses(headers, "Cc")
        + _addresses(headers, "Bcc"),
        timestamp=canonical_timestamp or parsed_date_header,
        date=parsed_date_header or raw_date_header,
        body=retrieval_body,
        snippet=retrieval_snippet,
        labels=message.get("labelIds", []) or [],
        internal_date=internal_date_ms,
        display_body=display_body,
        display_snippet=display_snippet,
    )


def _gmail_service():
    credentials = load_credentials()
    if credentials is None:
        raise PermissionError("Gmail is not connected. Connect Gmail first.")
    return build("gmail", "v1", credentials=credentials, cache_discovery=False)


def search_messages(
    query: str = "",
    *,
    max_messages: int = 10,
    page_size: int = 10,
    include_spam_trash: bool = False,
    label_ids: list[str] | None = None,
    service=None,
) -> tuple[list[NormalizedGmailMessage], bool]:
    """Search messages across Gmail API pages with query filters, labels, and spam/trash controls."""
    if max_messages <= 0 or page_size <= 0:
        raise ValueError("message and page sizes must be positive")
    gmail_service = service or _gmail_service()
    messages: list[NormalizedGmailMessage] = []
    page_token: str | None = None
    seen_tokens: set[str] = set()
    seen_message_ids: set[str] = set()
    complete = False
    while len(messages) < max_messages:
        request_size = min(page_size, max_messages - len(messages))
        list_kwargs = {"userId": "me", "maxResults": request_size}
        if page_token:
            list_kwargs["pageToken"] = page_token
        if query:
            list_kwargs["q"] = query
        if include_spam_trash:
            list_kwargs["includeSpamTrash"] = True
        if label_ids:
            list_kwargs["labelIds"] = label_ids

        request = gmail_service.users().messages().list(**list_kwargs)
        listing = request.execute()
        for item in listing.get("messages", []):
            if len(messages) >= max_messages:
                break
            msg_id = item.get("id", "")
            if not msg_id or msg_id in seen_message_ids:
                continue
            seen_message_ids.add(msg_id)
            message = (
                gmail_service.users()
                .messages()
                .get(userId="me", id=msg_id, format="full")
                .execute()
            )
            messages.append(normalize_message(message))
        next_token = listing.get("nextPageToken")
        if not next_token:
            complete = True
            break
        if next_token in seen_tokens or next_token == page_token:
            raise RuntimeError("Gmail pagination returned a repeated page token")
        seen_tokens.add(next_token)
        page_token = next_token
    return messages, complete


def fetch_messages_paginated(
    *,
    max_messages: int,
    page_size: int,
    service=None,
) -> tuple[list[NormalizedGmailMessage], bool]:
    """Fetch normalized messages across Gmail list pages without over-fetching."""
    return search_messages(
        query="",
        max_messages=max_messages,
        page_size=page_size,
        include_spam_trash=False,
        label_ids=None,
        service=service,
    )


def fetch_messages(limit: int = 10) -> list[NormalizedGmailMessage]:
    messages, _ = fetch_messages_paginated(
        max_messages=limit, page_size=min(limit, config.GMAIL_SYNC_PAGE_SIZE)
    )
    return messages