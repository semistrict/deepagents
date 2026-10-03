"""The agent in the app's process, on the durable runtime.

!!! warning "Experimental"

    Enabled with `DEEPAGENTS_CODE_DURABLE` and the `durable` extra; may change
    or be removed without notice.

The app normally builds the agent in a `langgraph dev` server and talks to it
over HTTP. In durable mode it builds the same agent here, with
`deepagents_durable` running the agent loop in place of a LangGraph graph,
and drives it directly as a local agent (see `DurableClient`). Each thread
is a durable session file of its own under `threads/` beside `sessions.db`,
opened by one process at a time, and listed through the index there. The
agent's store lives in memory, per process, as the server's does.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from deepagents_code._env_vars import DURABLE, is_env_truthy

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from deepagents_durable import Threads
    from langgraph.store.base import BaseStore

    from deepagents_code.client.durable_client import DurableClient

logger = logging.getLogger(__name__)

_threads: Threads | None = None
"""The threads this process opened."""
_store: BaseStore | None = None
"""The agents' store, shared by every agent this process builds."""


def durable_enabled() -> bool:
    """Report whether this process runs agents on the durable runtime.

    Returns:
        Whether `DEEPAGENTS_CODE_DURABLE` is set.
    """
    return is_env_truthy(DURABLE)


def durable_directory() -> Path:
    """Locate the thread files and their index.

    Returns:
        The `threads` directory beside `sessions.db`.
    """
    from deepagents_code.sessions import get_db_path

    return get_db_path().with_name("threads")


def durable_threads() -> Threads:
    """Return this process's threads, which open as agents use them.

    Returns:
        The process-wide view of the thread directory.
    """
    global _threads  # noqa: PLW0603  # one view of the directory per process
    if _threads is None:
        from deepagents_durable import Threads

        _threads = Threads(durable_directory())
    return _threads


def _durable_store() -> BaseStore:
    global _store  # noqa: PLW0603  # one store per process, as one server has one
    if _store is None:
        from langgraph.store.memory import InMemoryStore

        _store = InMemoryStore()
    return _store


async def close_durable_threads() -> None:
    """Close every thread this process opened.

    Their unfinished runs resume when the threads are next opened.
    """
    global _threads, _store
    if _threads is not None:
        threads, _threads, _store = _threads, None, None
        await threads.close()


async def start_durable_agent(
    *,
    cwd: str | None = None,
    auto_approve: bool = False,
    enable_shell: bool = True,
    host: str | None = None,
    port: int | None = None,
    **options: Any,  # `ServerConfig.from_cli_args` arguments
) -> tuple[DurableClient, None, None]:
    """Build the agent `start_server_and_get_agent` serves, on the durable runtime.

    Takes the same arguments. The server address is unused.

    Returns:
        `(client, None, None)`: there is no server process or MCP session
            manager to hand back.
    """
    del host, port
    from deepagents_durable import agent_factory

    from deepagents_code._server_config import ServerConfig
    from deepagents_code.client.durable_client import DurableClient
    from deepagents_code.client.launch.server_manager import (
        _capture_project_context,
        _preflight_validate_mcp_config,
    )
    from deepagents_code.project_utils import ProjectContext
    from deepagents_code.server_graph import _make_graphs

    project_context = (
        ProjectContext.from_user_cwd(Path(cwd))
        if cwd is not None
        else _capture_project_context()
    )
    _preflight_validate_mcp_config(
        mcp_config_path=options.get("mcp_config_path"),
        no_mcp=options.get("no_mcp", False),
    )
    config = ServerConfig.from_cli_args(
        project_context=project_context,
        auto_approve=auto_approve,
        enable_shell=enable_shell,
        **options,
    )
    threads, store = durable_threads(), _durable_store()
    runtime = await _make_graphs(
        config_override=config,
        project_context_override=project_context,
        agent_factory=agent_factory(threads, store=store),
    )
    logger.info("Agent runs in process on the durable runtime (%s)", threads.directory)
    return DurableClient(runtime.agent, store), None, None


@asynccontextmanager
async def durable_session(
    **options: Any,  # `start_durable_agent` arguments
) -> AsyncIterator[tuple[DurableClient, None]]:
    """Run `server_session`'s role on the durable runtime, closing threads on exit.

    Yields:
        `(client, None)`: there is no server process.
    """
    client, _, _ = await start_durable_agent(**options)
    try:
        yield client, None
    finally:
        await close_durable_threads()
