"""The app driving the real agent in process, on the durable runtime."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from deepagents_code._fake_models import _ToolBindingFakeModel
from deepagents_code._testing_models import (
    DCA_TEST_DELEGATE_WRITE_MARKER,
    DCA_TEST_WRITE_FILE_MARKER,
    SUBAGENT_WRITE_CONTENT,
    TOP_LEVEL_WRITE_CONTENT,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from langchain_core.messages import BaseMessage

    from deepagents_code._ask_user_types import AskUserWidgetResult, Question
    from deepagents_code.app import DeepAgentsApp
    from deepagents_code.client.durable_client import DurableClient

ASSISTANT = "itest-durable"

_PROVIDERS = """\
[models.providers.itest]
class_path = "deepagents_code._testing_models:ToolCallingIntegrationChatModel"
models = ["fake"]
[models.providers.asking]
class_path = "{module}:AskingModel"
models = ["fake"]
"""

_ROOMY = {"max_input_tokens": 1_000_000}
"""The fake models advertise 8k input tokens, less than dcode's prompt and tools."""


class AskingModel(_ToolBindingFakeModel):
    """Asks the user one question with `ask_user`, then repeats the answer."""

    disable_streaming: bool = True

    def _generate(self, messages: list[BaseMessage], *_: Any, **__: Any) -> ChatResult:
        results = [message for message in messages if message.type == "tool"]
        if results:
            message = AIMessage(content=f"you said: {results[-1].text}")
        else:
            question = {"question": "Favorite color?", "type": "text"}
            call = {
                "name": "ask_user",
                "args": {"questions": [question]},
                "id": "ask-1",
                "type": "tool_call",
            }
            message = AIMessage(content="", tool_calls=[call])
        return ChatResult(generations=[ChatGeneration(message=message)])


@pytest.fixture
async def project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[Path]:
    """An isolated home, state directory, and project, with the fake models."""
    from deepagents_code import model_config, sessions
    from deepagents_code.client.launch.durable import close_durable_threads
    from deepagents_code.config import create_model

    config_dir = tmp_path / "home" / ".deepagents"
    config_dir.mkdir(parents=True)
    (config_dir / "config.toml").write_text(_PROVIDERS.format(module=__name__))
    (tmp_path / "state").mkdir()
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("DEEPAGENTS_CODE_NO_UPDATE_CHECK", "1")
    monkeypatch.chdir(project)
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_DIR", config_dir)
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", config_dir / "config.toml")
    monkeypatch.setattr(sessions, "_db_path", tmp_path / "state" / "sessions.db")
    model_config.clear_caches()
    create_model("itest:fake").apply_to_runtime_state()
    yield project
    # The app keeps its threads open until its process ends.
    await close_durable_threads()
    model_config.clear_caches()


@asynccontextmanager
async def durable_agent(model: str = "itest:fake") -> AsyncIterator[DurableClient]:
    """The agent the app builds on the durable runtime; its threads close after."""
    from deepagents_code.client.launch.durable import (
        close_durable_threads,
        start_durable_agent,
    )

    agent, _, _ = await start_durable_agent(
        assistant_id=ASSISTANT,
        model_name=model,
        no_mcp=True,
        enable_shell=False,
        enable_ask_user=True,
        interactive=True,
        sandbox_type="none",
        sandbox_id=None,
        sandbox_snapshot_name=None,
        sandbox_setup=None,
        model_params=None,
        profile_overrides=_ROOMY,
        mcp_config_path=None,
        trust_project_mcp=None,
    )
    try:
        yield agent
    finally:
        await close_durable_threads()


def _server_kwargs() -> dict[str, Any]:
    """What `main` hands the app for an interactive session, with the fake model."""
    return {
        "assistant_id": ASSISTANT,
        "model_name": "itest:fake",
        "summarization_model": None,
        "model_params": None,
        "cli_max_retries": None,
        "profile_overrides": _ROOMY,
        "sandbox_type": "none",
        "sandbox_id": None,
        "sandbox_snapshot_name": None,
        "sandbox_setup": None,
        "enable_ask_user": True,
        "enable_interpreter": None,
        "interpreter_ptc": None,
        "interpreter_ptc_acknowledge_unsafe": False,
        "allow_fs_tools": None,
        "auto_classifier_model": None,
        "mcp_config_path": None,
        "no_mcp": True,
        "trust_project_mcp": None,
        "trust_project_extensions": False,
        "extension_paths": (),
        "interactive": True,
        "recursion_limit": None,
    }


class Recorder:
    """The adapter's UI callbacks: records what is asked, and answers it."""

    def __init__(self, decision: str, answer: str) -> None:
        self.decision = decision
        self.answer = answer
        self.approvals: list[list[dict[str, Any]]] = []
        self.questions: list[list[Question]] = []

    async def request_ask_user(
        self, questions: list[Question]
    ) -> asyncio.Future[AskUserWidgetResult]:
        self.questions.append(questions)
        answered: asyncio.Future[AskUserWidgetResult]
        answered = asyncio.get_running_loop().create_future()
        answered.set_result({"type": "answered", "answers": [self.answer]})
        return answered

    async def request_approval(
        self, action_requests: list[dict[str, Any]], _assistant_id: str | None
    ) -> asyncio.Future[object]:
        self.approvals.append(action_requests)
        answered = asyncio.get_running_loop().create_future()
        answered.set_result({"type": self.decision})
        return answered

    async def mount_message(self, _: object) -> bool:
        await asyncio.sleep(0)
        return True

    def update_status(self, _: str) -> None:
        return None


async def _turn(
    agent: DurableClient,
    prompt: str,
    thread_id: str,
    *,
    auto_approve: bool = True,
    decision: str = "approve",
    answer: str = "",
) -> Recorder:
    """Run one user turn the way the TUI does."""
    from deepagents_code.app import TextualSessionState
    from deepagents_code.tui.textual_adapter import (
        TextualUIAdapter,
        execute_task_textual,
    )

    recorder = Recorder(decision, answer)
    adapter = TextualUIAdapter(
        mount_message=recorder.mount_message,
        update_status=recorder.update_status,
        request_approval=recorder.request_approval,
        request_ask_user=recorder.request_ask_user,
    )
    await execute_task_textual(
        user_input=prompt,
        agent=agent,
        assistant_id=ASSISTANT,
        session_state=TextualSessionState(
            thread_id=thread_id, auto_approve=auto_approve
        ),
        adapter=adapter,
    )
    return recorder


async def _transcript(agent: DurableClient, thread_id: str) -> list[tuple[str, str]]:
    state = await agent.aget_state({"configurable": {"thread_id": thread_id}})
    # A thread with no run yet has no values, as on LangGraph.
    messages = state.values.get("messages", [])
    return [(message.type, message.text) for message in messages]


async def _settled(app: DeepAgentsApp, thread_id: str, messages: int) -> DurableClient:
    """Pump the app until its agent rests with `messages` messages on the thread.

    Returns:
        The app's agent client.
    """
    from deepagents_code.client.durable_client import DurableClient

    async def settled() -> DurableClient:
        while True:
            await asyncio.sleep(0.05)
            agent = app._agent
            if (
                isinstance(agent, DurableClient)
                and not app._agent_running
                and app._lc_thread_id == thread_id
                and len(await _transcript(agent, thread_id)) == messages
            ):
                return agent

    return await asyncio.wait_for(settled(), timeout=20)


@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    ("marker", "content", "result"),
    [
        (DCA_TEST_WRITE_FILE_MARKER, TOP_LEVEL_WRITE_CONTENT, "Updated file {}"),
        (DCA_TEST_DELEGATE_WRITE_MARKER, SUBAGENT_WRITE_CONTENT, "done"),
    ],
    ids=["top_level", "subagent"],
)
async def test_an_auto_approved_write_runs_without_asking(
    project: Path,
    marker: str,
    content: str,
    result: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A forked subagent sees the parent's delegate marker and delegates again.
    monkeypatch.setenv("DEEPAGENTS_CODE_FORKED_SUBAGENTS", "0")
    target = project / "out.txt"
    async with durable_agent() as agent:
        recorder = await _turn(agent, f"{marker}{target}", "auto")
        transcript = await _transcript(agent, "auto")
    assert recorder.approvals == []
    assert transcript[-2:] == [("tool", result.format(target)), ("ai", "done")]
    assert target.read_text() == content


@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    ("decision", "written"), [("approve", True), ("reject", False)]
)
async def test_a_manual_write_asks_and_follows_the_decision(
    project: Path, decision: str, *, written: bool
) -> None:
    target = project / "out.txt"
    prompt = f"{DCA_TEST_WRITE_FILE_MARKER}{target}"
    async with durable_agent() as agent:
        recorder = await _turn(
            agent, prompt, "manual", auto_approve=False, decision=decision
        )
        turn = await _transcript(agent, "manual")
        await _turn(agent, "thanks", "manual", auto_approve=False)
        after = await agent.aget_state({"configurable": {"thread_id": "manual"}})
    # A bare reject ends the turn without resuming it; the next message starts anew.
    ran = [("tool", f"Updated file {target}"), ("ai", "done")] if written else []
    assert turn == [("human", prompt), ("ai", ""), *ran]
    assert [(m.type, m.text) for m in after.values["messages"]][-2:] == [
        ("human", "thanks"),
        ("ai", "done"),
    ]
    assert after.next == ()
    assert [[request["name"] for request in asked] for asked in recorder.approvals] == [
        ["write_file"]
    ]
    assert target.exists() is written


@pytest.mark.timeout(120)
async def test_ask_user_asks_through_the_ui_and_the_tool_gets_the_answer(
    project: Path,
) -> None:
    from deepagents_code.config import create_model

    del project
    create_model("asking:fake").apply_to_runtime_state()
    async with durable_agent(model="asking:fake") as agent:
        recorder = await _turn(agent, "ask me", "asked", answer="teal")
        transcript = await _transcript(agent, "asked")
        state = await agent.aget_state({"configurable": {"thread_id": "asked"}})
    assert [[q["question"] for q in asked] for asked in recorder.questions] == [
        ["Favorite color?"]
    ]
    assert transcript == [
        ("human", "ask me"),
        ("ai", ""),
        ("tool", "Q: Favorite color?\nA: teal"),
        ("ai", "you said: Q: Favorite color?\nA: teal"),
    ]
    assert state.next == ()


@pytest.mark.timeout(120)
async def test_a_thread_is_read_back_after_reopening(project: Path) -> None:
    target = project / "out.txt"
    async with durable_agent() as agent:
        await _turn(agent, f"{DCA_TEST_WRITE_FILE_MARKER}{target}", "reopened")
    async with durable_agent() as agent:
        transcript = await _transcript(agent, "reopened")
    assert [kind for kind, _ in transcript] == ["human", "ai", "tool", "ai"]
    assert transcript[-1] == ("ai", "done")


@pytest.mark.timeout(120)
async def test_threads_are_discovered_from_the_durable_index(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from deepagents_code import sessions

    monkeypatch.setenv("DEEPAGENTS_CODE_DURABLE", "1")
    async with durable_agent() as agent:
        await _turn(agent, "remember this", "found")
    listed = await sessions.list_threads()
    assert [
        (row["thread_id"], row["agent_name"], row["cwd"], row["message_count"])
        for row in listed
    ] == [("found", ASSISTANT, str(project), 2)]
    assert [row["initial_prompt"] for row in listed] == ["remember this"]
    assert await sessions.get_most_recent(ASSISTANT) == "found"
    assert await sessions.thread_exists("found")
    assert not await sessions.thread_exists("lost")
    assert await sessions.get_thread_cwd("found") == str(project)
    assert await sessions.delete_thread("found")
    assert await sessions.list_threads() == []


@pytest.mark.timeout(120)
async def test_the_app_chats_with_an_agent_running_in_process(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from deepagents_code.app import DeepAgentsApp

    monkeypatch.setenv("DEEPAGENTS_CODE_DURABLE", "1")
    app = DeepAgentsApp(
        assistant_id=ASSISTANT,
        thread_id="app-thread",
        cwd=project,
        initial_prompt="hello there",
        server_kwargs=_server_kwargs(),
    )
    async with app.run_test():
        agent = await _settled(app, "app-thread", messages=2)
        transcript = await _transcript(agent, "app-thread")
    assert transcript == [("human", "hello there"), ("ai", "done")]


@pytest.mark.timeout(120)
async def test_the_app_resumes_the_most_recent_thread(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from deepagents_code.app import DeepAgentsApp

    monkeypatch.setenv("DEEPAGENTS_CODE_DURABLE", "1")
    async with durable_agent() as agent:
        await _turn(agent, "from before", "earlier")
    app = DeepAgentsApp(
        assistant_id=ASSISTANT,
        cwd=project,
        resume_thread="__MOST_RECENT__",
        initial_prompt="and now",
        server_kwargs=_server_kwargs(),
    )
    async with app.run_test():
        agent = await _settled(app, "earlier", messages=4)
        transcript = await _transcript(agent, "earlier")
    assert transcript == [
        ("human", "from before"),
        ("ai", "done"),
        ("human", "and now"),
        ("ai", "done"),
    ]


@pytest.mark.timeout(120)
async def test_the_interpreter_is_not_snapshotted_into_the_thread(
    project: Path,
) -> None:
    del project
    async with durable_agent() as agent:
        await _turn(agent, "hello", "unsnapshotted")
        state = await agent.aget_state({"configurable": {"thread_id": "unsnapshotted"}})
    assert "_quickjs_snapshot_payload" not in state.values
