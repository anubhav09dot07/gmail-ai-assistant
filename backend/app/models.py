from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class NormalizedGmailMessage:
    message_id: str
    thread_id: str
    subject: str
    sender: str
    recipients: list[str]
    timestamp: str
    date: str
    body: str           # retrieval body — stored in Qdrant, never modified
    snippet: str        # Gmail-provided snippet — used for retrieval context
    labels: list[str]
    internal_date: int = 0
    display_body: str = ""    # display-only cleaned body — never stored in Qdrant
    display_snippet: str = "" # display-only cleaned snippet

    def to_api_dict(self, body_limit: int = 4000) -> dict:
        data = asdict(self)
        # For API responses use the display body when available, truncated for preview
        display = self.display_body or self.body
        data["body"] = display[:body_limit] if display else None
        data["display_body"] = display[:body_limit] if display else None
        data["display_snippet"] = self.display_snippet or self.snippet
        return data