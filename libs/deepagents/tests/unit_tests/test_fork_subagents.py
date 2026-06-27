from __future__ import annotations

from typing import Any

import pytest
from langchain.tools import ToolRuntime
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable

from deepagents.graph import create_deep_agent
from deepagents.middleware.subagents import (
    ALL_FORKED_USAGE_GUIDANCE,
    FORK_USAGE_GUIDANCE,
    FORKED_SUBAGENT_MARKER,
    _build_task_tool,
)
from tests.unit_tests.chat_model import GenericFakeChatModel


class _RecordingChatModel(BaseChatModel):
    recorded_calls: list[list[Any]] = []  # noqa: RUF012  # pydantic field, per-instance
    scripted_task_description: str = "inspect the target"
    _call_idx: int = 0

    @property
    def _llm_type(self) -> str:
        return "recording"

    def _generate(
        self,
        messages: list[Any],
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.recorded_calls.append(list(messages))
        idx = self._call_idx
        self._call_idx += 1
        if idx == 0:
            message = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "task",
                        "args": {
                            "description": self.scripted_task_description,
                            "subagent_type": "forked",
                        },
                        "id": "call-fork-1",
                        "type": "tool_call",
                    }
                ],
            )
        else:
            message = AIMessage(content="fork final response")
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:  # type: ignore[override]  # noqa: ANN401
        return self


class _RecordingRunnable(Runnable):
    def __init__(self, response_text: str = "done") -> None:
        self.state_inputs: list[dict[str, Any]] = []
        self.configs: list[dict[str, Any] | None] = []
        self._response_text = response_text

    def invoke(self, input: dict[str, Any], config: dict[str, Any] | None = None, **_: Any) -> dict[str, Any]:  # noqa: A002
        self.state_inputs.append(input)
        self.configs.append(config)
        return {"messages": [AIMessage(content=self._response_text)]}

    async def ainvoke(self, input: dict[str, Any], config: dict[str, Any] | None = None, **_: Any) -> dict[str, Any]:  # noqa: A002
        return self.invoke(input, config)


def _runtime(
    *,
    state: dict[str, Any],
    tool_call_id: str = "tc-1",
    config: dict[str, Any] | None = None,
) -> ToolRuntime:
    return ToolRuntime(
        state=state,
        context=None,
        tool_call_id=tool_call_id,
        store=None,
        stream_writer=lambda _: None,
        config=config or {},
    )


def test_fork_prepends_parent_messages_when_seeding_state() -> None:
    fork_runnable = _RecordingRunnable()
    plain_runnable = _RecordingRunnable()
    task_tool = _build_task_tool(
        [
            {"name": "forked", "description": "Fork worker.", "runnable": fork_runnable, "fork": True},
            {"name": "plain", "description": "Plain worker.", "runnable": plain_runnable},
        ]
    )

    parent_messages = [HumanMessage(content="parent Q"), AIMessage(content="parent A")]
    task_tool.func(description="do thing", subagent_type="forked", runtime=_runtime(state={"messages": parent_messages}))
    task_tool.func(description="do other", subagent_type="plain", runtime=_runtime(state={"messages": parent_messages}, tool_call_id="tc-2"))

    assert [message.content for message in fork_runnable.state_inputs[0]["messages"]] == ["parent Q", "parent A", "do thing"]
    assert [message.content for message in plain_runnable.state_inputs[0]["messages"]] == ["do other"]
    assert fork_runnable.configs[0]["configurable"]["ls_agent_type"] == "fork-subagent"
    assert plain_runnable.configs[0]["configurable"]["ls_agent_type"] == "subagent"


def test_fork_derives_child_thread_id_from_parent_thread_and_tool_call() -> None:
    fork_runnable = _RecordingRunnable()
    task_tool = _build_task_tool([{"name": "forked", "description": "Fork worker.", "runnable": fork_runnable, "fork": True}])

    task_tool.func(
        description="do thing",
        subagent_type="forked",
        runtime=_runtime(
            state={"messages": []},
            tool_call_id="call-1",
            config={"configurable": {"thread_id": "parent-thread"}},
        ),
    )

    assert fork_runnable.configs[0]["configurable"]["thread_id"] == "parent-thread:subagent:call-1"


def test_fork_derives_child_thread_id_without_parent_thread() -> None:
    fork_runnable = _RecordingRunnable()
    task_tool = _build_task_tool([{"name": "forked", "description": "Fork worker.", "runnable": fork_runnable, "fork": True}])

    task_tool.func(
        description="do thing",
        subagent_type="forked",
        runtime=_runtime(state={"messages": []}, tool_call_id="call-1"),
    )

    assert fork_runnable.configs[0]["configurable"]["thread_id"] == "subagent:call-1"


def test_nonfork_does_not_derive_child_thread_id() -> None:
    plain_runnable = _RecordingRunnable()
    task_tool = _build_task_tool([{"name": "plain", "description": "Plain worker.", "runnable": plain_runnable}])

    task_tool.func(
        description="do thing",
        subagent_type="plain",
        runtime=_runtime(
            state={"messages": []},
            tool_call_id="call-1",
            config={"configurable": {"thread_id": "parent-thread"}},
        ),
    )

    assert "thread_id" not in plain_runnable.configs[0]["configurable"]


def test_fork_drops_current_task_tool_call_from_seeded_state() -> None:
    fork_runnable = _RecordingRunnable()
    task_tool = _build_task_tool([{"name": "forked", "description": "Fork worker.", "runnable": fork_runnable, "fork": True}])
    messages = [
        HumanMessage(content="parent Q"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "task",
                    "args": {"description": "do thing", "subagent_type": "forked"},
                    "id": "tc-1",
                    "type": "tool_call",
                }
            ],
        ),
    ]

    task_tool.func(description="do thing", subagent_type="forked", runtime=_runtime(state={"messages": messages}))

    assert [message.content for message in fork_runnable.state_inputs[0]["messages"]] == ["parent Q", "do thing"]


def test_fork_preserves_prior_tool_history_when_trimming_current_task() -> None:
    fork_runnable = _RecordingRunnable()
    task_tool = _build_task_tool([{"name": "forked", "description": "Fork worker.", "runnable": fork_runnable, "fork": True}])
    prior_tool_call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "search",
                "args": {"query": "cacheable context"},
                "id": "prior-search",
                "type": "tool_call",
            }
        ],
    )
    prior_tool_result = ToolMessage(content="prior search result", tool_call_id="prior-search")
    messages = [
        HumanMessage(content="Earlier user context."),
        prior_tool_call,
        prior_tool_result,
        AIMessage(content="Earlier assistant answer."),
        HumanMessage(content="Current request."),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "task",
                    "args": {"description": "Do forked work", "subagent_type": "forked"},
                    "id": "call-fork",
                    "type": "tool_call",
                }
            ],
        ),
    ]

    task_tool.func(
        description="Do forked work",
        subagent_type="forked",
        runtime=_runtime(state={"messages": messages}, tool_call_id="call-fork"),
    )

    forked = fork_runnable.state_inputs[0]["messages"]
    assert forked[:-1] == messages[:-1]
    assert forked[-1].content == "Do forked work"
    assert prior_tool_call in forked
    assert prior_tool_result in forked


def test_fork_trims_current_tool_message_if_ai_turn_is_absent() -> None:
    fork_runnable = _RecordingRunnable()
    task_tool = _build_task_tool([{"name": "forked", "description": "Fork worker.", "runnable": fork_runnable, "fork": True}])
    messages = [
        HumanMessage(content="Earlier user context."),
        AIMessage(content="Earlier assistant answer."),
        ToolMessage(content="current task result bookkeeping", tool_call_id="call-fork"),
    ]

    task_tool.func(
        description="Do forked work",
        subagent_type="forked",
        runtime=_runtime(state={"messages": messages}, tool_call_id="call-fork"),
    )

    assert [message.content for message in fork_runnable.state_inputs[0]["messages"]] == [
        "Earlier user context.",
        "Earlier assistant answer.",
        "Do forked work",
    ]


async def test_async_fork_matches_sync_state_and_thread_id_behavior() -> None:
    fork_runnable = _RecordingRunnable()
    task_tool = _build_task_tool([{"name": "forked", "description": "Fork worker.", "runnable": fork_runnable, "fork": True}])

    await task_tool.coroutine(
        description="Do async forked work",
        subagent_type="forked",
        runtime=_runtime(
            state={
                "messages": [
                    HumanMessage(content="Async parent context."),
                    AIMessage(content="Async prior answer."),
                ]
            },
            tool_call_id="call-async-fork",
            config={"configurable": {"thread_id": "parent-thread"}},
        ),
    )

    assert [message.content for message in fork_runnable.state_inputs[0]["messages"]] == [
        "Async parent context.",
        "Async prior answer.",
        "Do async forked work",
    ]
    assert fork_runnable.configs[0]["configurable"]["thread_id"] == "parent-thread:subagent:call-async-fork"


def test_fork_injects_subagent_prompt_into_trailing_task_message() -> None:
    fork_runnable = _RecordingRunnable()
    task_tool = _build_task_tool(
        [
            {
                "name": "forked",
                "description": "Fork worker.",
                "runnable": fork_runnable,
                "fork": True,
                "subagent_system_prompt": "Use careful analysis.",
            }
        ]
    )

    task_tool.func(description="inspect file", subagent_type="forked", runtime=_runtime(state={"messages": []}))

    assert fork_runnable.state_inputs[0]["messages"][-1].content == "Use careful analysis.\n\ninspect file"


def test_task_tool_guidance_for_mixed_and_all_forked_subagents() -> None:
    mixed_tool = _build_task_tool(
        [
            {"name": "forked", "description": "Fork worker.", "runnable": _RecordingRunnable(), "fork": True},
            {"name": "plain", "description": "Plain worker.", "runnable": _RecordingRunnable()},
        ]
    )
    all_forked_tool = _build_task_tool(
        [
            {"name": "alpha", "description": "First fork.", "runnable": _RecordingRunnable(), "fork": True},
            {"name": "beta", "description": "Second fork.", "runnable": _RecordingRunnable(), "fork": True},
        ]
    )

    assert f"- forked {FORKED_SUBAGENT_MARKER}: Fork worker." in mixed_tool.description
    assert FORK_USAGE_GUIDANCE in mixed_tool.description
    assert "- alpha: First fork." in all_forked_tool.description
    assert FORKED_SUBAGENT_MARKER not in all_forked_tool.description
    assert ALL_FORKED_USAGE_GUIDANCE in all_forked_tool.description


def test_create_deep_agent_rejects_model_on_forked_subagent() -> None:
    with pytest.raises(ValueError, match="cannot declare a model"):
        create_deep_agent(
            model=GenericFakeChatModel(messages=iter([])),
            subagents=[
                {
                    "name": "forked",
                    "description": "Fork worker.",
                    "system_prompt": "Work carefully.",
                    "model": GenericFakeChatModel(messages=iter([])),
                    "fork": True,
                }
            ],
        )


def test_create_deep_agent_rejects_compiled_forked_subagent() -> None:
    with pytest.raises(ValueError, match="CompiledSubAgent"):
        create_deep_agent(
            model=GenericFakeChatModel(messages=iter([])),
            subagents=[
                {
                    "name": "compiled",
                    "description": "Compiled worker.",
                    "runnable": _RecordingRunnable(),
                    "fork": True,
                }
            ],
        )


def test_create_deep_agent_fork_inherits_parent_prompt_and_message_prefix() -> None:
    model = _RecordingChatModel()
    agent = create_deep_agent(
        model=model,
        system_prompt="PARENT_PROMPT_PREFIX",
        subagents=[
            {
                "name": "forked",
                "description": "Fork worker.",
                "system_prompt": "FORK_ROLE_PROMPT",
                "tools": [],
                "fork": True,
            }
        ],
    )

    agent.invoke({"messages": [HumanMessage(content="MAIN_USER_MSG")]})

    assert len(model.recorded_calls) >= 2
    parent_call = model.recorded_calls[0]
    fork_call = model.recorded_calls[1]
    parent_system = next(message.content for message in parent_call if isinstance(message, SystemMessage))
    fork_system = next(message.content for message in fork_call if isinstance(message, SystemMessage))

    assert fork_system == parent_system
    assert [message.content for message in fork_call if isinstance(message, HumanMessage)] == [
        "MAIN_USER_MSG",
        "You are running as a forked subagent named `forked`. "
        "The system prompt above was inherited verbatim from the parent agent to "
        "preserve prompt-cache reuse; it may mention capabilities that do not apply "
        "to you. Your actual environment:\n"
        "\n- Your declared tools: (none - rely on built-in filesystem/todo tools)"
        "\n- You do NOT have the `task` tool. You cannot spawn further subagents."
        "\n\nYour role as this subagent:\n"
        "FORK_ROLE_PROMPT\n\ninspect the target",
    ]
