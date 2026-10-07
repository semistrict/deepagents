"""Thread discovery when agents run on the durable runtime.

!!! warning "Experimental"

    See `client.launch.durable`.

`sessions` finds threads in the LangGraph checkpoint tables. On the durable
runtime each thread is a session file of its own, and the threads' index
already holds what discovery shows: the latest run's metadata (agent, cwd,
git branch), the message count, and the first message, kept current as runs
end. These functions answer `sessions`' queries from that index, never
opening a thread's file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from deepagents_code._constants import DEFAULT_THREAD_LIMIT
from deepagents_code.client.launch.durable import durable_threads

if TYPE_CHECKING:
    from deepagents_durable import ThreadSummary

    from deepagents_code.sessions import ThreadInfo


def _info(row: ThreadSummary) -> ThreadInfo:
    from deepagents_code.sessions import ThreadInfo

    metadata = row.metadata
    return ThreadInfo(
        thread_id=row.thread_id,
        agent_name=metadata.get("agent_name"),
        updated_at=row.updated_at,
        created_at=row.created_at,
        git_branch=metadata.get("git_branch"),
        cwd=metadata.get("cwd"),
        # The index row changes whenever the thread does.
        latest_checkpoint_id=row.updated_at,
        message_count=row.message_count,
        initial_prompt=row.first_message,
    )


async def list_threads(
    agent_name: str | None = None,
    limit: int = DEFAULT_THREAD_LIMIT,
    include_message_count: bool = False,  # noqa: ARG001  # always included
    sort_by: str = "updated",
    branch: str | None = None,
    cwd: str | None = None,
) -> list[ThreadInfo]:
    """List indexed threads; see `sessions.list_threads`.

    Returns:
        Matching threads, newest first by `sort_by`.

    Raises:
        ValueError: If `sort_by` is not `"updated"` or `"created"`.
    """
    from deepagents_code.sessions import _cache_recent_threads

    if sort_by not in {"updated", "created"}:
        msg = f"Invalid sort_by {sort_by!r}; expected 'updated' or 'created'"
        raise ValueError(msg)
    rows = [
        _info(row)
        for row in await durable_threads().recent()
        if (agent_name is None or row.metadata.get("agent_name") == agent_name)
        and (branch is None or row.metadata.get("git_branch") == branch)
        and (cwd is None or row.metadata.get("cwd") == cwd)
    ]
    if sort_by == "created":
        rows.sort(key=lambda row: row.get("created_at") or "", reverse=True)
    threads = rows[:limit]
    if sort_by == "updated" and branch is None and cwd is None:
        _cache_recent_threads(agent_name, limit, threads)
    return threads


async def populate_thread_checkpoint_details(
    threads: list[ThreadInfo],
    *,
    include_message_count: bool = True,
    include_initial_prompt: bool = True,
) -> list[ThreadInfo]:
    """Fill message counts and first prompts from the index; see `sessions`.

    Returns:
        The same list, populated in place.
    """
    for thread in threads:
        row = await durable_threads().get(thread["thread_id"])
        if row is None:
            continue
        if include_message_count:
            thread["message_count"] = row.message_count
        if include_initial_prompt:
            thread["initial_prompt"] = row.first_message
    return threads


async def prewarm_thread_message_counts(limit: int | None = None) -> None:
    """Cache the recent threads for the selector; see `sessions`."""
    from deepagents_code.sessions import get_thread_limit

    thread_limit = limit if limit is not None else get_thread_limit()
    if thread_limit >= 1:
        await list_threads(limit=thread_limit)


async def get_most_recent(
    agent_name: str | None = None,
    *,
    exclude_thread_id: str | None = None,
) -> str | None:
    """The most recently updated thread; see `sessions.get_most_recent`.

    Returns:
        Its thread ID, or `None` when no thread matches.
    """
    for row in await durable_threads().recent():
        if row.thread_id == exclude_thread_id:
            continue
        if agent_name is None or row.metadata.get("agent_name") == agent_name:
            return row.thread_id
    return None


async def get_thread_updated_at(thread_id: str) -> str | None:
    """When a thread was last updated.

    Returns:
        The ISO timestamp, or `None` for an unknown thread.
    """
    row = await durable_threads().get(thread_id)
    return None if row is None else row.updated_at


async def get_thread_agent(thread_id: str) -> str | None:
    """The agent of a thread's latest run.

    Returns:
        The agent name, or `None` for an unknown thread.
    """
    row = await durable_threads().get(thread_id)
    return None if row is None else row.metadata.get("agent_name")


async def get_thread_cwd(thread_id: str) -> str | None:
    """The working directory of a thread's latest run.

    Returns:
        The directory, or `None` when unknown.
    """
    row = await durable_threads().get(thread_id)
    value = None if row is None else row.metadata.get("cwd")
    return value if isinstance(value, str) and value else None


async def thread_exists(thread_id: str) -> bool:  # noqa: RUF029  # same signature as `sessions`
    """Whether a thread has a session file.

    Returns:
        `True` when the thread exists.
    """
    return durable_threads().exists(thread_id)


async def find_similar_threads(thread_id: str, limit: int = 3) -> list[str]:
    """Indexed threads whose IDs start with `thread_id`.

    Returns:
        Up to `limit` matching IDs, in order.
    """
    rows = await durable_threads().recent()
    return sorted(row.thread_id for row in rows if row.thread_id.startswith(thread_id))[
        :limit
    ]


async def delete_thread(thread_id: str) -> bool:
    """Delete a thread's file, index row, and offloaded history.

    Returns:
        Whether the thread existed.
    """
    from deepagents_code.offload import delete_offloaded_history
    from deepagents_code.sessions import _recent_threads_cache

    deleted = await durable_threads().delete(thread_id)
    if deleted:
        for key, rows in list(_recent_threads_cache.items()):
            _recent_threads_cache[key] = [
                row for row in rows if row["thread_id"] != thread_id
            ]
    delete_offloaded_history(thread_id)
    return deleted
