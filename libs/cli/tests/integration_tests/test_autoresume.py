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

        # All phases run against the same server session so the in-memory
        # runtime sees the thread copy (matching the real autofork flow).
        async with server_session(
            assistant_id=assistant_id,
            model_name="itest:fake",
            no_mcp=True,
            enable_shell=False,
            interactive=True,
            sandbox_type="none",
        ) as (agent, _server_proc):
            # Phase 1: create a persisted thread with a few turns.
            for i in range(1, 4):
                await _run_turn(
                    agent,
                    thread_id=source_thread_id,
                    assistant_id=assistant_id,
                    prompt=f"Turn {i}: tell me something interesting.",
                )

            # Phase 2: fork via the server's /threads/<id>/copy API.
            graph = agent._get_graph()
            client = graph._validate_client()
            resp = await client.http.post(f"/threads/{source_thread_id}/copy", json={})
            forked_thread_id = resp["thread_id"]
            assert forked_thread_id != source_thread_id

            # Phase 3: stream against the forked thread in the same server.
            await _run_turn(
                agent,
                thread_id=forked_thread_id,
                assistant_id=assistant_id,
                prompt="[SYSTEM] This session was auto-forked. "
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


@pytest.mark.timeout(180)
async def test_fork_across_server_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fork a thread from a prior server session and stream in a new one.

    This is the real user flow: the source thread was created in a previous
    ``deepagents`` invocation (different server process). The new server's
    in-memory runtime has no knowledge of it — only the SQLite checkpointer
    has the data.  The fork must ensure the thread is loaded from the
    persister before calling /threads/<id>/copy.

    This test would have caught the 409 Conflict bug where /copy failed
    because the source thread didn't exist in the new runtime.
    """
    home_dir = tmp_path / "home"
    project_dir = tmp_path / "project"
    assistant_id = "itest-fork-restart"

    home_dir.mkdir()
    project_dir.mkdir()

    monkeypatch.setenv("HOME", str(home_dir))
    monkeypatch.setenv("DEEPAGENTS_CLI_NO_UPDATE_CHECK", "1")
    monkeypatch.chdir(project_dir)

    _write_model_config(home_dir)

    from deepagents_cli import model_config
    from deepagents_cli.config import create_model
    from deepagents_cli.server_manager import server_session
    from deepagents_cli.sessions import generate_thread_id, thread_exists

    config_path = home_dir / ".deepagents" / "config.toml"
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_DIR", config_path.parent)
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", config_path)

    model_config.clear_caches()
    try:
        create_model("itest:fake").apply_to_settings()
        source_thread_id = generate_thread_id()

        # Server 1: create a thread and shut down.
        async with server_session(
            assistant_id=assistant_id,
            model_name="itest:fake",
            no_mcp=True,
            enable_shell=False,
            interactive=True,
            sandbox_type="none",
        ) as (agent, _):
            await _run_turn(
                agent,
                thread_id=source_thread_id,
                assistant_id=assistant_id,
                prompt="Remember this number: 42",
            )

        assert await thread_exists(source_thread_id)

        # Server 2: fresh process — source thread only in SQLite.
        async with server_session(
            assistant_id=assistant_id,
            model_name="itest:fake",
            no_mcp=True,
            enable_shell=False,
            interactive=True,
            sandbox_type="none",
        ) as (agent, _):
            graph = agent._get_graph()
            client = graph._validate_client()

            # Replicate the _fork_via_server logic: ensure thread is loaded,
            # then copy.
            try:
                await client.threads.get(source_thread_id)
            except Exception:
                await client.threads.create(
                    thread_id=source_thread_id, if_exists="do_nothing"
                )
            resp = await client.http.post(
                f"/threads/{source_thread_id}/copy", json={}
            )
            forked_thread_id = resp["thread_id"]
            assert forked_thread_id != source_thread_id

            # Stream against the fork in the new server.
            await _run_turn(
                agent,
                thread_id=forked_thread_id,
                assistant_id=assistant_id,
                prompt="hi",
            )
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
    from deepagents_cli.sessions import generate_thread_id

    config_path = home_dir / ".deepagents" / "config.toml"
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_DIR", config_path.parent)
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", config_path)

    model_config.clear_caches()
    try:
        create_model(_OPENROUTER_MODEL).apply_to_settings()
        source_thread_id = generate_thread_id()

        async with server_session(
            assistant_id=assistant_id,
            model_name=_OPENROUTER_MODEL,
            no_mcp=True,
            enable_shell=False,
            interactive=True,
            sandbox_type="none",
        ) as (agent, _server_proc):
            # Phase 1: seed a thread with one real turn.
            await _run_turn(
                agent,
                thread_id=source_thread_id,
                assistant_id=assistant_id,
                prompt="Say hello in one word.",
            )

            # Phase 2: fork via server API.
            graph = agent._get_graph()
            client = graph._validate_client()
            resp = await client.http.post(f"/threads/{source_thread_id}/copy", json={})
            forked_thread_id = resp["thread_id"]

            # Phase 3: stream against the fork.
            await _run_turn(
                agent,
                thread_id=forked_thread_id,
                assistant_id=assistant_id,
                prompt="What did I just ask you?",
            )
    finally:
        model_config.clear_caches()
