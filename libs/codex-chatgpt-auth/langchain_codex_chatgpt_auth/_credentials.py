"""Read ChatGPT OAuth credentials from Codex's ``auth.json``."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CHATGPT_BASE_URL = "https://chatgpt.com/backend-api/codex"
"""Base URL for the ChatGPT backend API used with Codex OAuth tokens.

The OpenAI SDK appends ``/responses`` to this, resulting in
``https://chatgpt.com/backend-api/codex/responses``.
"""

_CODEX_HOME_ENV = "CODEX_HOME"
"""Environment variable to override the default ``~/.codex`` directory."""


def _codex_home() -> Path:
    """Resolve the Codex home directory.

    Respects the ``CODEX_HOME`` environment variable, falling back to
    ``~/.codex``.
    """
    codex_home = os.environ.get(_CODEX_HOME_ENV)
    if codex_home:
        return Path(codex_home)
    return Path.home() / ".codex"


def _codex_auth_path() -> Path:
    """Resolve the path to Codex's ``auth.json``.

    Returns:
        Path to the ``auth.json`` file (may not exist).
    """
    return _codex_home() / "auth.json"


@dataclass(frozen=True)
class ChatGPTCredentials:
    """ChatGPT OAuth credentials extracted from Codex's ``auth.json``."""

    access_token: str
    """OAuth access token (used as Bearer token for API calls)."""

    account_id: str
    """ChatGPT account/workspace ID (sent as ``chatgpt-account-id`` header)."""


def get_chatgpt_credentials() -> ChatGPTCredentials | None:
    """Read ChatGPT OAuth credentials from Codex's ``auth.json``.

    Looks for ``$CODEX_HOME/auth.json`` (defaulting to ``~/.codex/auth.json``)
    and returns credentials only when ``auth_mode`` is ``chatgpt`` and the
    required token fields are present.

    Returns:
        Credentials if available, ``None`` otherwise.
    """
    auth_path = _codex_auth_path()
    if not auth_path.is_file():
        return None

    try:
        data = json.loads(auth_path.read_text())
    except (OSError, json.JSONDecodeError):
        logger.debug("Failed to read %s", auth_path)
        return None

    if data.get("auth_mode") != "chatgpt":
        return None

    tokens = data.get("tokens")
    if not isinstance(tokens, dict):
        return None

    access_token = tokens.get("access_token")
    account_id = tokens.get("account_id") or data.get("account_id")
    if not access_token or not account_id:
        return None

    return ChatGPTCredentials(access_token=access_token, account_id=account_id)


def get_openai_kwargs() -> dict[str, Any] | None:
    """Build ``langchain-openai`` kwargs from ChatGPT credentials.

    If Codex has valid ChatGPT OAuth tokens, returns a dict suitable for
    passing to ``ChatOpenAI(...)`` or ``init_chat_model("openai:...", ...)``:

    - ``api_key``: the OAuth access token
    - ``base_url``: the ChatGPT backend API URL
    - ``default_headers``: includes ``chatgpt-account-id``

    Returns:
        Kwargs dict if credentials are available, ``None`` otherwise.

    Example::

        from langchain_codex_chatgpt_auth import get_openai_kwargs

        kwargs = get_openai_kwargs()
        if kwargs:
            model = ChatOpenAI(model="gpt-5.2", **kwargs)
    """
    creds = get_chatgpt_credentials()
    if creds is None:
        return None
    model_kwargs: dict[str, Any] = {"instructions": " "}
    if not os.environ.get("DEEPAGENTS_CHATGPT_NO_PRIORITY_TIER"):
        model_kwargs["service_tier"] = "priority"

    return {
        "api_key": creds.access_token,
        "base_url": CHATGPT_BASE_URL,
        "default_headers": {"chatgpt-account-id": creds.account_id},
        "use_responses_api": True,
        "streaming": True,
        "store": False,
        "model_kwargs": model_kwargs,
    }


def strip_reasoning(messages: list[Any]) -> list[Any]:
    """Strip reasoning content blocks from AI messages.

    The ChatGPT backend with ``store=false`` cannot look up reasoning item
    IDs from previous responses.  Removing these blocks from the conversation
    history prevents ``NotFoundError`` on follow-up requests.

    Mutates messages in-place for efficiency and returns the same list.

    Args:
        messages: List of LangChain message objects.

    Returns:
        The same list with reasoning blocks removed from AI messages.
    """
    for msg in messages:
        content = getattr(msg, "content", None)
        if isinstance(content, list):
            cleaned = [
                c for c in content
                if not (isinstance(c, dict) and c.get("type") == "reasoning")
            ]
            if len(cleaned) != len(content):
                # Use object.__setattr__ to bypass frozen dataclass if needed.
                try:
                    msg.content = cleaned  # type: ignore[attr-defined]
                except (AttributeError, TypeError):
                    object.__setattr__(msg, "content", cleaned)
    return messages


_MODELS_ENDPOINT = f"{CHATGPT_BASE_URL}/models"
"""Endpoint for listing available ChatGPT models."""


def get_chatgpt_models() -> list[str]:
    """Fetch available model slugs from the ChatGPT backend API.

    Calls ``GET /codex/models`` with the same credentials used for chat.
    Returns visible models sorted by priority (highest first).
    Falls back to an empty list on any error.

    Returns:
        List of model slug strings.
    """
    import urllib.request  # noqa: PLC0415
    import urllib.error  # noqa: PLC0415

    creds = get_chatgpt_credentials()
    if creds is None:
        return []

    req = urllib.request.Request(
        f"{_MODELS_ENDPOINT}?client_version=1.0.0",
        headers={
            "Authorization": f"Bearer {creds.access_token}",
            "chatgpt-account-id": creds.account_id,
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, OSError, json.JSONDecodeError, TimeoutError):
        logger.debug("Failed to fetch models from %s", _MODELS_ENDPOINT)
        return []

    models = data.get("models") if isinstance(data, dict) else None
    if not isinstance(models, list):
        return []

    visible = [
        m for m in models
        if isinstance(m, dict)
        and m.get("visibility") == "list"
        and m.get("slug")
    ]
    visible.sort(key=lambda m: m.get("priority", 999))
    return [m["slug"] for m in visible]
