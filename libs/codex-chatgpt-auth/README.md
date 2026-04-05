# langchain-codex-chatgpt-auth

Use ChatGPT OAuth credentials from [Codex](https://github.com/openai/codex) with `langchain-openai`.

If you've authenticated with `codex login` using a ChatGPT account, this package reads the stored OAuth tokens and returns kwargs suitable for `ChatOpenAI`.

## Usage

```python
from langchain_codex_chatgpt_auth import get_openai_kwargs
from langchain_openai import ChatOpenAI

kwargs = get_openai_kwargs()
if kwargs:
    model = ChatOpenAI(model="gpt-5.2", **kwargs)
```

## How it works

Reads `~/.codex/auth.json` (or `$CODEX_HOME/auth.json`) and, when `auth_mode` is `"chatgpt"`, extracts the OAuth `access_token` and `account_id`. These are returned as:

- `api_key`: the OAuth access token (used as Bearer token)
- `base_url`: `https://chatgpt.com/backend-api`
- `default_headers`: includes `chatgpt-account-id`

This package has **zero dependencies** -- it only reads a JSON file from disk.
