"""Deep Agents backends whose state lives in the durable thread."""

from __future__ import annotations

from typing import Any

from deepagents.backends import StateBackend

from deepagents_durable.scope import CURRENT


class DurableStateBackend(StateBackend):
    """Files in the thread's `files` field, written with the step that wrote them.

    `StateBackend` reaches the field through LangGraph's channel reads and
    writes; this one reads and writes the current step's scope instead. A
    write is visible to later reads in the same step and lands in the thread
    atomically with the step's result.
    """

    def _read_files(self) -> dict[str, Any]:
        return self._scope().read("files") or {}

    def _send_files_update(self, update: dict[str, Any]) -> None:
        self._scope().send([("files", update)])

    @staticmethod
    def _scope() -> Any:  # noqa: ANN401  # the current step's scope
        scope = CURRENT.get()
        if scope is None:
            msg = "DurableStateBackend can only read and write files inside a running agent step."
            raise RuntimeError(msg)
        return scope
