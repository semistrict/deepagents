"""Tests for ChatGPT credential reading from Codex auth.json."""

import json

from unittest.mock import patch

from langchain_codex_chatgpt_auth import (
    ChatGPTCredentials,
    get_chatgpt_credentials,
    get_chatgpt_models,
    get_openai_kwargs,
)
from langchain_codex_chatgpt_auth._credentials import CHATGPT_BASE_URL


class TestGetChatGPTCredentials:
    """Tests for get_chatgpt_credentials()."""

    def test_returns_credentials_for_chatgpt_mode(self, tmp_path, monkeypatch):
        auth = {
            "auth_mode": "chatgpt",
            "OPENAI_API_KEY": None,
            "tokens": {
                "access_token": "eyJ-access",
                "refresh_token": "rt_refresh",
                "account_id": "acct-123",
            },
        }
        auth_path = tmp_path / "auth.json"
        auth_path.write_text(json.dumps(auth))
        monkeypatch.setenv("CODEX_HOME", str(tmp_path))

        creds = get_chatgpt_credentials()

        assert creds == ChatGPTCredentials(
            access_token="eyJ-access",
            account_id="acct-123",
        )

    def test_returns_none_for_apikey_mode(self, tmp_path, monkeypatch):
        auth = {
            "auth_mode": "apikey",
            "OPENAI_API_KEY": "sk-test",
            "tokens": None,
        }
        (tmp_path / "auth.json").write_text(json.dumps(auth))
        monkeypatch.setenv("CODEX_HOME", str(tmp_path))

        assert get_chatgpt_credentials() is None

    def test_returns_none_when_no_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CODEX_HOME", str(tmp_path))

        assert get_chatgpt_credentials() is None

    def test_returns_none_for_invalid_json(self, tmp_path, monkeypatch):
        (tmp_path / "auth.json").write_text("{bad json")
        monkeypatch.setenv("CODEX_HOME", str(tmp_path))

        assert get_chatgpt_credentials() is None

    def test_returns_none_when_tokens_missing(self, tmp_path, monkeypatch):
        auth = {"auth_mode": "chatgpt", "tokens": None}
        (tmp_path / "auth.json").write_text(json.dumps(auth))
        monkeypatch.setenv("CODEX_HOME", str(tmp_path))

        assert get_chatgpt_credentials() is None

    def test_returns_none_when_access_token_missing(self, tmp_path, monkeypatch):
        auth = {
            "auth_mode": "chatgpt",
            "tokens": {"refresh_token": "rt", "account_id": "acct-123"},
        }
        (tmp_path / "auth.json").write_text(json.dumps(auth))
        monkeypatch.setenv("CODEX_HOME", str(tmp_path))

        assert get_chatgpt_credentials() is None

    def test_returns_none_when_account_id_missing(self, tmp_path, monkeypatch):
        auth = {
            "auth_mode": "chatgpt",
            "tokens": {"access_token": "eyJ-access"},
        }
        (tmp_path / "auth.json").write_text(json.dumps(auth))
        monkeypatch.setenv("CODEX_HOME", str(tmp_path))

        assert get_chatgpt_credentials() is None

    def test_uses_default_codex_home_when_env_unset(self, tmp_path, monkeypatch):
        monkeypatch.delenv("CODEX_HOME", raising=False)
        monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

        auth = {
            "auth_mode": "chatgpt",
            "tokens": {"access_token": "tok", "account_id": "acct"},
        }
        codex_dir = tmp_path / ".codex"
        codex_dir.mkdir()
        (codex_dir / "auth.json").write_text(json.dumps(auth))

        creds = get_chatgpt_credentials()

        assert creds is not None
        assert creds.access_token == "tok"


class TestGetOpenAIKwargs:
    """Tests for get_openai_kwargs()."""

    def test_returns_kwargs_with_priority_tier_by_default(self, tmp_path, monkeypatch):
        auth = {
            "auth_mode": "chatgpt",
            "tokens": {"access_token": "eyJ-tok", "account_id": "acct-456"},
        }
        (tmp_path / "auth.json").write_text(json.dumps(auth))
        monkeypatch.setenv("CODEX_HOME", str(tmp_path))
        monkeypatch.delenv("DEEPAGENTS_CHATGPT_NO_PRIORITY_TIER", raising=False)

        kwargs = get_openai_kwargs()

        assert kwargs == {
            "api_key": "eyJ-tok",
            "base_url": CHATGPT_BASE_URL,
            "default_headers": {"chatgpt-account-id": "acct-456"},
            "use_responses_api": True,
            "streaming": True,
            "store": False,
            "model_kwargs": {"instructions": " ", "service_tier": "priority"},
        }

    def test_no_priority_tier_when_env_set(self, tmp_path, monkeypatch):
        auth = {
            "auth_mode": "chatgpt",
            "tokens": {"access_token": "eyJ-tok", "account_id": "acct-456"},
        }
        (tmp_path / "auth.json").write_text(json.dumps(auth))
        monkeypatch.setenv("CODEX_HOME", str(tmp_path))
        monkeypatch.setenv("DEEPAGENTS_CHATGPT_NO_PRIORITY_TIER", "1")

        kwargs = get_openai_kwargs()

        assert kwargs == {
            "api_key": "eyJ-tok",
            "base_url": CHATGPT_BASE_URL,
            "default_headers": {"chatgpt-account-id": "acct-456"},
            "use_responses_api": True,
            "streaming": True,
            "store": False,
            "model_kwargs": {"instructions": " "},
        }

    def test_returns_none_when_no_credentials(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CODEX_HOME", str(tmp_path))

        assert get_openai_kwargs() is None


class TestGetChatGPTModels:
    """Tests for get_chatgpt_models()."""

    def test_returns_models_sorted_by_priority(self, tmp_path, monkeypatch):
        auth = {
            "auth_mode": "chatgpt",
            "tokens": {"access_token": "tok", "account_id": "acct"},
        }
        (tmp_path / "auth.json").write_text(json.dumps(auth))
        monkeypatch.setenv("CODEX_HOME", str(tmp_path))

        api_response = json.dumps({
            "models": [
                {"slug": "gpt-b", "visibility": "list", "priority": 5},
                {"slug": "gpt-a", "visibility": "list", "priority": 1},
                {"slug": "gpt-hidden", "visibility": "hide", "priority": 2},
            ],
        }).encode()

        import io
        import urllib.request

        def fake_urlopen(*_args, **_kwargs):
            resp = io.BytesIO(api_response)
            resp.__enter__ = lambda s: s
            resp.__exit__ = lambda s, *a: None
            return resp

        with patch.object(urllib.request, "urlopen", fake_urlopen):
            models = get_chatgpt_models()

        assert models == ["gpt-a", "gpt-b"]

    def test_returns_empty_on_no_credentials(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CODEX_HOME", str(tmp_path))

        assert get_chatgpt_models() == []

    def test_returns_empty_on_network_error(self, tmp_path, monkeypatch):
        auth = {
            "auth_mode": "chatgpt",
            "tokens": {"access_token": "tok", "account_id": "acct"},
        }
        (tmp_path / "auth.json").write_text(json.dumps(auth))
        monkeypatch.setenv("CODEX_HOME", str(tmp_path))

        import urllib.request

        with patch.object(
            urllib.request, "urlopen", side_effect=OSError("connection failed")
        ):
            assert get_chatgpt_models() == []
