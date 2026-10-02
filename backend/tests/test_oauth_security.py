import stat
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from backend.app.oauth import save_credentials


class OAuthSecurityTests(unittest.TestCase):
    def test_saved_token_file_is_owner_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            token_path = Path(directory) / "token.json"
            credentials = SimpleNamespace(to_json=lambda: '{"token":"redacted"}')
            with patch("backend.app.oauth.config.TOKEN_FILE", token_path):
                save_credentials(credentials)
            self.assertEqual(stat.S_IMODE(token_path.stat().st_mode), 0o600)


if __name__ == "__main__":
    unittest.main()