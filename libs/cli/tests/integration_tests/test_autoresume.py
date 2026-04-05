"""Integration test: fork a real persisted thread and stream against the fork."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

_OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "")

# Use OpenAI provider pointed at OpenRouter's base URL with a cheap model.
_OPENROUTER_MODEL = "openai:openai/gpt-4.1-nano"
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _write_model_config(home_dir: Path) -> None:
    """Write a temp config that points the server subprocess at the test model."""
    config_dir = home_dir / ".deepagents"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "config.toml").write_text(
        """
[models.providers.itest]
class_path = "deepagents_cli._testing_models:DeterministicIntegrationChatModel"
models = ["fake"]
""".strip()
        + "\n"
    )


async def _run_turn(agent, *, thread_id: str, assistant_id: str, prompt: str) -> None:
    """Execute one real remote agent turn and drain the stream to completion."""
    from deepagents_cli.config import build_stream_config

    config = build_stream_config(thread_id, assistant_id)
    stream_input = {"messages": [{"role": "user", "content": prompt}]}
    async for _chunk in agent.astream(
        stream_input,
        stream_mode=["messages", "updates"],
        subgraphs=True,
        config=config,
        durability="exit",
    ):
        pass


@pytest.mark.timeout(180)
async def test_fork_thread_and_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fork a persisted thread and successfully stream against the forked copy.

    This reproduces the BadRequestError seen when auto-resume forks a
    checkpoint and the agent tries to continue the conversation.
    """
    home_dir = tmp_path / "home"
    project_dir = tmp_path / "project"
    assistant_id = "itest-fork"

    home_dir.mkdir()
    project_dir.mkdir()

    monkeypatch.setenv("HOME", str(home_dir))
    monkeypatch.setenv("DEEPAGENTS_CLI_NO_UPDATE_CHECK", "1")
    monkeypatch.chdir(project_dir)

    _write_model_config(home_dir)

    from deepagents_cli import model_config
    from deepagents_cli.config import create_model
    from deepagents_cli.server_manager import server_session
    from deepagents_cli.sessions import (
        fork_thread,
        generate_thread_id,
        thread_exists,
    )

    config_path = home_dir / ".deepagents" / "config.toml"
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_DIR", config_path.parent)
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", config_path)

    model_config.clear_caches()
    try:
        create_model("itest:fake").apply_to_settings()
        source_thread_id = generate_thread_id()

        # Phase 1: create a real persisted thread with a few turns.
        async with server_session(
            assistant_id=assistant_id,
            model_name="itest:fake",
            no_mcp=True,
            enable_shell=False,
            interactive=True,
            sandbox_type="none",
        ) as (agent, _server_proc):
            for i in range(1, 4):
                await _run_turn(
                    agent,
                    thread_id=source_thread_id,
                    assistant_id=assistant_id,
                    prompt=f"Turn {i}: tell me something interesting.",
                )

        assert await thread_exists(source_thread_id)

        # Phase 2: fork the thread.
        forked_thread_id = await fork_thread(source_thread_id)
        assert forked_thread_id is not None
        assert forked_thread_id != source_thread_id
        assert await thread_exists(forked_thread_id)

        # Phase 3: stream a new turn against the forked thread.
        # This is where the BadRequestError occurred.
        async with server_session(
            assistant_id=assistant_id,
            model_name="itest:fake",
            no_mcp=True,
            enable_shell=False,
            interactive=True,
            sandbox_type="none",
        ) as (agent, _server_proc):
            # This should NOT raise.
            await _run_turn(
                agent,
                thread_id=forked_thread_id,
                assistant_id=assistant_id,
                prompt="[SYSTEM] This session was auto-resumed. "
                "Treat this as a new session but carry forward context.",
            )
            await _run_turn(
                agent,
                thread_id=forked_thread_id,
                assistant_id=assistant_id,
                prompt="hi",
            )

        # Source thread should be unchanged.
        assert await thread_exists(source_thread_id)
    finally:
        model_config.clear_caches()


@pytest.mark.timeout(120)
@pytest.mark.skipif(not _OPENROUTER_KEY, reason="OPENROUTER_API_KEY not set")
async def test_fork_thread_and_stream_openrouter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fork a thread created with an OpenRouter model and stream against it.

    Verifies that the forked checkpoint's message history (which includes
    system messages from the agent prompt) is accepted by the real LLM API.
    Skipped when OPENROUTER_API_KEY is not set.
    """
    home_dir = tmp_path / "home"
    project_dir = tmp_path / "project"
    assistant_id = "itest-fork-or"

    home_dir.mkdir()
    project_dir.mkdir()

    monkeypatch.setenv("HOME", str(home_dir))
    monkeypatch.setenv("DEEPAGENTS_CLI_NO_UPDATE_CHECK", "1")
    # Use OpenRouter via the OpenAI provider by setting the base URL + key.
    monkeypatch.setenv("OPENAI_API_KEY", _OPENROUTER_KEY)
    monkeypatch.setenv("OPENAI_BASE_URL", _OPENROUTER_BASE_URL)
    monkeypatch.chdir(project_dir)

    _write_model_config(home_dir)

    from deepagents_cli import model_config
    from deepagents_cli.config import create_model
    from deepagents_cli.server_manager import server_session
    from deepagents_cli.sessions import (
        fork_thread,
        generate_thread_id,
        thread_exists,
    )

    config_path = home_dir / ".deepagents" / "config.toml"
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_DIR", config_path.parent)
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", config_path)

    model_config.clear_caches()
    try:
        create_model(_OPENROUTER_MODEL).apply_to_settings()
        source_thread_id = generate_thread_id()

        # Phase 1: seed a thread with one real turn.
        async with server_session(
            assistant_id=assistant_id,
            model_name=_OPENROUTER_MODEL,
            no_mcp=True,
            enable_shell=False,
            interactive=True,
            sandbox_type="none",
        ) as (agent, _server_proc):
            await _run_turn(
                agent,
                thread_id=source_thread_id,
                assistant_id=assistant_id,
                prompt="Say hello in one word.",
            )

        assert await thread_exists(source_thread_id)

        # Phase 2: fork.
        forked_thread_id = await fork_thread(source_thread_id)
        assert forked_thread_id is not None

        # Phase 3: stream against the fork — this is where the
        # "System messages are not allowed" error occurred.
        async with server_session(
            assistant_id=assistant_id,
            model_name=_OPENROUTER_MODEL,
            no_mcp=True,
            enable_shell=False,
            interactive=True,
            sandbox_type="none",
        ) as (agent, _server_proc):
            await _run_turn(
                agent,
                thread_id=forked_thread_id,
                assistant_id=assistant_id,
                prompt="What did I just ask you?",
            )
    finally:
        model_config.clear_caches()
