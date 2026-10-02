from pathlib import Path

from google_auth_oauthlib.flow import Flow

from . import config


def ensure_oauth_configured() -> None:
    if not config.GOOGLE_CLIENT_ID or not config.GOOGLE_CLIENT_SECRET:
        raise RuntimeError(
            "Google OAuth is not configured. Set GOOGLE_CLIENT_ID and "
            "GOOGLE_CLIENT_SECRET in backend/.env."
        )


def create_flow() -> Flow:
    ensure_oauth_configured()
    client_config = {
        "web": {
            "client_id": config.GOOGLE_CLIENT_ID,
            "client_secret": config.GOOGLE_CLIENT_SECRET,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": [config.GOOGLE_REDIRECT_URI],
        }
    }
    flow = Flow.from_client_config(client_config, scopes=config.GOOGLE_SCOPES)
    flow.redirect_uri = config.GOOGLE_REDIRECT_URI
    return flow


def save_credentials(credentials) -> None:
    config.TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    config.TOKEN_FILE.write_text(credentials.to_json(), encoding="utf-8")
    config.TOKEN_FILE.chmod(0o600)


def load_credentials():
    if not config.TOKEN_FILE.is_file():
        return None
    from google.oauth2.credentials import Credentials

    return Credentials.from_authorized_user_file(
        str(config.TOKEN_FILE), config.GOOGLE_SCOPES
    )


def clear_credentials() -> None:
    Path(config.TOKEN_FILE).unlink(missing_ok=True)