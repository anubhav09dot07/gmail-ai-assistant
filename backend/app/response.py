from .models import NormalizedGmailMessage


def format_message(message: NormalizedGmailMessage) -> dict:
    return message.to_api_dict()