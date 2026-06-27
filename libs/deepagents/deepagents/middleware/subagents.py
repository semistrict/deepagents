"""Middleware for providing subagents to an agent via a `task` tool."""

import contextlib
import dataclasses
import json
from collections.abc import Awaitable, Callable, Generator, Sequence
from typing import Any, NotRequired, TypedDict, cast

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, InterruptOnConfig
from langchain.agents.middleware.types import (
    AgentMiddleware,
    ContextT,
    ModelRequest,
    ModelResponse,
    ResponseT,
)
from langchain.agents.structured_output import ResponseFormat
from langchain.tools import BaseTool, ToolRuntime
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import StructuredTool
from langgraph.types import Command
from langsmith.run_helpers import get_tracing_context, tracing_context
from pydantic import BaseModel, Field

from deepagents.backends.protocol import BackendFactory, BackendProtocol
from deepagents.middleware._utils import append_to_system_message
from deepagents.middleware.filesystem import FilesystemPermission

SUBAGENT_RESPONSE_FORMAT_CONFIG_KEY = "__deepagents_subagent_response_format"
"""Configurable key used by task-tool callers to request dynamic response format."""


class SubAgent(TypedDict):
    """Specification for an agent.

    When using `create_deep_agent`, subagents automatically receive
    a default middleware stack before any custom `middleware` specified in
    this spec.

    Required fields:
        name: Unique identifier for the subagent.

            The main agent uses this name when calling the `task()` tool.
        description: What this subagent does.

            Be specific and action-oriented. The main agent uses this
            to decide when to delegate.
        system_prompt: Instructions for the subagent.

            Include tool usage guidance and output format requirements.

    Optional fields:
        tools: Tools the subagent can use.

            If not specified, inherits tools from the main agent
            via `default_tools`.
        model: Override the main agent's model.

            Use the format `'provider:model-name'` (e.g., `'openai:gpt-5.5'`).
        middleware: Additional middleware for custom behavior, logging,
            or rate limiting.
        interrupt_on: Configure human-in-the-loop for specific tools.

            Requires a checkpointer.
        skills: Skill source paths for `SkillsMiddleware`.

            List of paths to skill directories
            (e.g., `["/skills/user/", "/skills/project/"]`).
        permissions: Filesystem permission rules for this subagent.

            If omitted, inherits the parent agent's permissions. If provided,
            replaces the parent agent's rules entirely for this subagent.

            Rules are evaluated in declaration order; the first match wins.
    """

    name: str
    """Unique identifier for the subagent."""

    description: str
    """What this subagent does.

    The main agent uses this to decide when to delegate.
    """

    system_prompt: str
    """Instructions for the subagent."""

    tools: NotRequired[Sequence[BaseTool | Callable | dict[str, Any]]]
    """Tools the subagent can use.

    If not specified, inherits from main agent.
    """

    model: NotRequired[str | BaseChatModel]
    """Override the main agent's model.

    Use `'provider:model-name'` format.
    """

    fork: NotRequired[bool]
    """Whether to fork the parent agent's context into the subagent.

    When `True`, the subagent inherits the parent agent's model, system prompt,
    and full message prefix so provider prompt caches can reuse the parent's
    already-seen input. The subagent's own `system_prompt` is not placed in
    the system slot; it is prepended to the trailing task `HumanMessage` so
    the inherited prefix stays byte-identical.

    Forked subagents cannot declare `model`; they always inherit the parent's
    model. `CompiledSubAgent` does not support fork mode because compiled
    runnables own their graph and system prompt.
    """

    middleware: NotRequired[list[AgentMiddleware]]
    """Additional middleware for custom behavior."""

    interrupt_on: NotRequired[dict[str, bool | InterruptOnConfig]]
    """Configure human-in-the-loop for specific tools."""

    skills: NotRequired[list[str]]
    """Skill source paths for `SkillsMiddleware`."""

    permissions: NotRequired[list[FilesystemPermission]]
    """List of `FilesystemPermission` rules for this subagent.

    If omitted, inherits the parent agent's permissions. If specified, replaces
    the parent's permissions entirely for this subagent.

    Rules are evaluated in declaration order; the first match wins.
    `FilesystemMiddleware` enforces these rules for the built-in filesystem
    tools on the subagent stack.
    """

    response_format: NotRequired[ResponseFormat[Any] | type | dict[str, Any]]
    """Structured output response format for the subagent.

    When specified, the subagent will produce a `structured_response` conforming
    to the given schema. The structured response is JSON-serialized and returned
    as the `ToolMessage` content to the parent agent, replacing the default
    last-message extraction.

    Accepted formats (from `langchain.agents.structured_output`):

    - `ToolStrategy(schema)`: Use tool calling to extract structured output from the model.
    - `ProviderStrategy(schema)`: Use the model provider's native structured output mode.
    - `AutoStrategy(schema)`: Automatically select the best strategy.
    - A bare Python `type`: A Pydantic `BaseModel` subclass, `dataclass`,
        or `TypedDict` class.

        Equivalent to `AutoStrategy(schema)`.
    - `dict[str, Any]`: A JSON schema dictionary
        (e.g., `{"type": "object", "properties": {...}, "required": [...]}`).

    Example:
        ```python
        from pydantic import BaseModel

        class Findings(BaseModel):
            findings: str
            confidence: float

        analyzer: SubAgent = {
            "name": "analyzer",
            "description": "Analyzes data and returns structured findings",
            "system_prompt": "Analyze the data and return your findings.",
            "model": "openai:gpt-5.5",
            "tools": [],
            "response_format": Findings,
        }
        ```
    """

    subagent_system_prompt: NotRequired[str]
    """Internal fork-mode role prompt injected into the trailing task message."""


class CompiledSubAgent(TypedDict):
    """A pre-compiled agent spec.

    !!! note

        The `runnable`'s state schema must include a 'messages' key.

        This is required for the subagent to communicate results back to
        the main agent.

    !!! note

        `CompiledSubAgent` runnables are used as provided. They do not
        inherit `create_deep_agent(state_schema=...)`; if the runnable
        needs custom state fields, compile it with a compatible state
        schema yourself.

    When the subagent completes, the parent reads the returned state:
    if `structured_response` is non-`None`, it is JSON-serialized and used as
    the `ToolMessage` content; otherwise, the last non-empty `AIMessage`
    text is used.

    Examples:
        Using `create_agent` with `response_format`:

        ```python
        from pydantic import BaseModel
        from langchain.agents import create_agent


        class Findings(BaseModel):
            summary: str
            confidence: float


        researcher: CompiledSubAgent = {
            "name": "researcher",
            "description": "Researches a topic and returns findings.",
            "runnable": create_agent(
                "openai:gpt-5.5",
                tools=[],  # your tools here
                response_format=Findings,
            ),
        }
        ```

        Custom `langgraph` graph (write `structured_response` directly):

        ```python
        def node(state):
            return {
                "messages": [...],
                "structured_response": Findings(summary="...", confidence=0.9),
            }
        ```
    """

    name: str
    """Unique identifier for the subagent."""

    description: str
    """What this subagent does.

    The main agent uses this to decide when to delegate.
    """

    runnable: Runnable
    """A custom agent implementation.

    Create a custom agent using either:

    1. LangChain's [`create_agent()`](https://docs.langchain.com/oss/python/langchain/quickstart)
    2. A custom graph using [`langgraph`](https://docs.langchain.com/oss/python/langgraph/quickstart)

    If you're creating a custom graph, make sure the state schema includes
    a 'messages' key. This is required for the subagent to communicate
    results back to the main agent.
    """

    fork: NotRequired[bool]
    """Unsupported. Setting this to `True` raises `ValueError`."""

    subagent_system_prompt: NotRequired[str]
    """Internal fork-mode role prompt injected into the trailing task message."""


DEFAULT_SUBAGENT_PROMPT = """In order to complete the objective that the user asks of you, you have access to a number of standard tools.

The calling agent only sees your final assistant message, not your intermediate work, tool results, or status tracking. Ensure your final
response contains the complete answer."""

_EXCLUDED_STATE_KEYS = {
    "messages",
    "todos",
    "structured_response",
}
"""State keys that are excluded when passing state to subagents and when
returning updates from subagents.

When returning updates:

1. The messages key is handled explicitly to ensure only the final message
    is included
2. The todos and `structured_response` keys are excluded as they do not have
    a defined reducer and no clear meaning for returning them from a subagent
    to the main agent.
3. Agent-private fields on middleware state schemas are excluded from both
    subagent output and subagent inputs.
"""

_FORK_INHERITED_STATE_KEYS: frozenset[str] = frozenset()
"""Forks inherit only these parent state keys in addition to `messages`."""


class TaskToolSchema(BaseModel):
    """Input schema for the `task` tool."""

    description: str = Field(
        description=(
            "A detailed description of the task for the subagent to perform autonomously. "
            "Include all necessary context and specify the expected output format."
        )
    )

    subagent_type: str = Field(description=("The type of subagent to use. Must be one of the available agent types listed in the tool description."))


TASK_TOOL_DESCRIPTION = """Launch an ephemeral subagent to handle complex, multi-step independent tasks with isolated context windows.

Available agent types and the tools they have access to:
{available_agents}

When using the Task tool, you must specify a subagent_type parameter to select which agent type to use.

## Usage notes:
1. Launch multiple agents concurrently whenever possible, to maximize performance; to do that, use a single message with multiple tool uses
2. When the agent is done, it will return a single message back to you. The result returned by the agent is not visible to the user. To show the user the result, you should send a text message back to the user with a concise summary of the result.
3. Each agent invocation is stateless. You will not be able to send additional messages to the agent, nor will the agent be able to communicate with you outside of its final report. Therefore, your prompt should contain a highly detailed task description for the agent to perform autonomously and you should specify exactly what information the agent should return back to you in its final and only message to you.
4. The agent's outputs should generally be trusted
5. Clearly tell the agent whether you expect it to create content, perform analysis, or just do research (search, file reads, web fetches, etc.), since it is not aware of the user's intent
6. If the agent description mentions that it should be used proactively, then you should try your best to use it without the user having to ask for it first. Use your judgement.
7. When only the general-purpose agent is provided, you should use it for all tasks. It is great for isolating context and token usage, and completing specific, complex tasks, as it has all the same capabilities as the main agent.

### Example usage of the general-purpose agent:

<example_agent_descriptions>
"general-purpose": use this agent for general purpose tasks, it has access to all tools as the main agent.
</example_agent_descriptions>

<example>
User: "I want to conduct research on the accomplishments of Lebron James, Michael Jordan, and Kobe Bryant, and then compare them."
Assistant: *Uses the task tool in parallel to conduct isolated research on each of the three players*
Assistant: *Synthesizes the results of the three isolated research tasks and responds to the User*
<commentary>
Research is a complex, multi-step task in it of itself.
The research of each individual player is not dependent on the research of the other players.
The assistant uses the task tool to break down the complex objective into three isolated tasks.
Each research task only needs to worry about context and tokens about one player, then returns synthesized information about each player as the Tool Result.
This means each research task can dive deep and spend tokens and context deeply researching each player, but the final result is synthesized information, and saves us tokens in the long run when comparing the players to each other.
</commentary>
</example>

<example>
User: "Analyze a single large code repository for security vulnerabilities and generate a report."
Assistant: *Launches a single `task` subagent for the repository analysis*
Assistant: *Receives report and integrates results into final summary*
<commentary>
Subagent is used to isolate a large, context-heavy task, even though there is only one. This prevents the main thread from being overloaded with details.
If the user then asks followup questions, we have a concise report to reference instead of the entire history of analysis and tool calls, which is good and saves us time and money.
</commentary>
</example>

<example>
User: "Schedule two meetings for me and prepare agendas for each."
Assistant: *Calls the task tool in parallel to launch two `task` subagents (one per meeting) to prepare agendas*
Assistant: *Returns final schedules and agendas*
<commentary>
Tasks are simple individually, but subagents help silo agenda preparation.
Each subagent only needs to worry about the agenda for one meeting.
</commentary>
</example>

<example>
User: "I want to order a pizza from Dominos, order a burger from McDonald's, and order a salad from Subway."
Assistant: *Calls tools directly in parallel to order a pizza from Dominos, a burger from McDonald's, and a salad from Subway*
<commentary>
The assistant did not use the task tool because the objective is super simple and clear and only requires a few trivial tool calls.
It is better to just complete the task directly and NOT use the `task` tool.
</commentary>
</example>

### Example usage with custom agents:

<example_agent_descriptions>
"content-reviewer": use this agent after you are done creating significant content or documents
"greeting-responder": use this agent when to respond to user greetings with a friendly joke
"research-analyst": use this agent to conduct thorough research on complex topics
</example_agent_descriptions>

<example>
user: "Please write a function that checks if a number is prime"
assistant: Sure let me write a function that checks if a number is prime
assistant: First let me use the Write tool to write a function that checks if a number is prime
assistant: I'm going to use the Write tool to write the following code:
<code>
function isPrime(n) {{
  if (n <= 1) return false
  for (let i = 2; i * i <= n; i++) {{
    if (n % i === 0) return false
  }}
  return true
}}
</code>
<commentary>
Since significant content was created and the task was completed, now use the content-reviewer agent to review the work
</commentary>
assistant: Now let me use the content-reviewer agent to review the code
assistant: Uses the Task tool to launch with the content-reviewer agent
</example>

<example>
user: "Can you help me research the environmental impact of different renewable energy sources and create a comprehensive report?"
<commentary>
This is a complex research task that would benefit from using the research-analyst agent to conduct thorough analysis
</commentary>
assistant: I'll help you research the environmental impact of renewable energy sources. Let me use the research-analyst agent to conduct comprehensive research on this topic.
assistant: Uses the Task tool to launch with the research-analyst agent, providing detailed instructions about what research to conduct and what format the report should take
</example>

<example>
user: "Hello"
<commentary>
Since the user is greeting, use the greeting-responder agent to respond with a friendly joke
</commentary>
assistant: "I'm going to use the Task tool to launch with the greeting-responder agent"
</example>"""  # noqa: E501

TASK_SYSTEM_PROMPT = """## `task` (subagent spawner)

You have access to a `task` tool to launch short-lived subagents that handle isolated tasks. These agents are ephemeral — they live only for the duration of the task and return a single result.

When to use the task tool:

- When a task is complex and multi-step, and can be fully delegated in isolation
- When a task is independent of other tasks and can run in parallel
- When a task requires focused reasoning or heavy token/context usage that would bloat the orchestrator thread
- When sandboxing improves reliability (e.g. code execution, structured searches, data formatting)
- When you only care about the output of the subagent, and not the intermediate steps (ex. performing a lot of research and then returned a synthesized report, performing a series of computations or lookups to achieve a concise, relevant answer.)

Subagent lifecycle:

1. **Spawn** → Provide clear role, instructions, and expected output
2. **Run** → The subagent completes the task autonomously
3. **Return** → The subagent provides a single structured result
4. **Reconcile** → Incorporate or synthesize the result into the main thread

When NOT to use the task tool:

- If you need to see the intermediate reasoning or steps after the subagent has completed (the task tool hides them)
- If the task is trivial (a few tool calls or simple lookup)
- If delegating does not reduce token usage, complexity, or context switching
- If splitting would add latency without benefit

## Important Task Tool Usage Notes to Remember

- Whenever possible, parallelize the work that you do. This is true for both tool_calls, and for tasks. Whenever you have independent steps to complete - make tool_calls, or kick off tasks (subagents) in parallel to accomplish them faster. This saves time for the user, which is incredibly important.
- Remember to use the `task` tool to silo independent tasks within a multi-part objective.
- You should use the `task` tool whenever you have a complex task that will take multiple steps, and is independent from other tasks that the agent needs to complete. These agents are highly competent and efficient."""  # noqa: E501

FORKED_SUBAGENT_MARKER = "[forked - inherits full conversation context]"
"""Annotation rendered next to forked subagents in mixed task-tool lists."""

FORK_USAGE_GUIDANCE = (
    "\n\n### Forked subagents\n"
    f"Subagents marked `{FORKED_SUBAGENT_MARKER}` inherit the full "
    "conversation context. Pass only the task-specific instruction in "
    "`description`; do not restate shared context already visible in the "
    "conversation. Non-forked subagents are stateless and still require full "
    "context in `description`."
)
"""Guidance block appended when only some subagents are forked."""

ALL_FORKED_USAGE_GUIDANCE = (
    "\n\n### Subagent context\n"
    "Subagents inherit the full conversation context, so only a minimal "
    "task-specific instruction is needed in `description`."
)
"""Guidance block appended when every subagent is forked."""


DEFAULT_GENERAL_PURPOSE_DESCRIPTION = "General-purpose agent for researching complex questions, searching for files and content, and executing multi-step tasks. When you are searching for a keyword or file and are not confident that you will find the right match in the first few tries use this agent to perform the search for you. This agent has access to all tools as the main agent."  # noqa: E501

GENERAL_PURPOSE_SUBAGENT: SubAgent = {
    "name": "general-purpose",
    "description": DEFAULT_GENERAL_PURPOSE_DESCRIPTION,
    "system_prompt": DEFAULT_SUBAGENT_PROMPT,
}
"""Base spec for general-purpose subagent (caller adds model, tools, middleware)."""


@contextlib.contextmanager
def _subagent_tracing_context(agent_type: str = "subagent") -> Generator[None, None, None]:
    """Context manager that tags subagent runs with `ls_agent_type`.

    Sets `ls_agent_type` on the langsmith tracing context `metadata`, which is
    propagated to LangSmith runs. This mirrors
    langchain's `ls_agent_type="root"` tagging behavior.

    Forwards all other current tracing-context fields (parent, client, tags,
    etc.) unchanged so this wrapper does not clobber the enclosing context.
    """
    current = get_tracing_context()

    merged_metadata = {**(current.get("metadata") or {}), "ls_agent_type": agent_type}
    # Pass every field from the current tracing context through to
    # `tracing_context` so we don't accidentally clobber fields that may be
    # added to langsmith in the future. The only change is `metadata`.

    kwargs: dict[str, Any] = {**current, "metadata": merged_metadata}

    with tracing_context(**kwargs):
        yield


class _SystemPromptPaddingMiddleware(AgentMiddleware[Any, Any, Any]):
    """Append deterministic text to a fork's inherited system prompt."""

    def __init__(self, *, text: str, name: str) -> None:
        super().__init__()
        self._text = text
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        return handler(request.override(system_message=append_to_system_message(request.system_message, self._text)))

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        return await handler(request.override(system_message=append_to_system_message(request.system_message, self._text)))


def _build_subagent_system_prompt(
    subagents: Sequence[SubAgent | CompiledSubAgent],
    system_prompt: str | None = TASK_SYSTEM_PROMPT,
) -> str | None:
    """Build the system prompt block appended by `SubAgentMiddleware`."""
    if not system_prompt or not subagents:
        return system_prompt
    agents_desc = "\n".join(f"- {s['name']}: {s['description']}" for s in subagents)
    return system_prompt + "\n\nAvailable subagent types:\n\n" + agents_desc


def _build_fork_subagent_system_prompt(subagents: Sequence[SubAgent | CompiledSubAgent]) -> str:
    """Build the task-tool prompt block used to align forked subagent cache."""
    system_prompt = _build_subagent_system_prompt(subagents)
    if system_prompt is None:
        msg = "Forked subagent cache alignment requires a subagent system prompt."
        raise ValueError(msg)
    return system_prompt


def _system_message_text(system_prompt: str | SystemMessage) -> str:
    """Flatten a string or text-only `SystemMessage` for inherited fork prompts."""
    if isinstance(system_prompt, str):
        return system_prompt
    parts: list[str] = []
    for block in system_prompt.content_blocks:
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def _tool_name(tool: BaseTool | Callable | dict[str, Any]) -> str | None:
    """Return a declared tool name for fork role guidance."""
    name = getattr(tool, "name", None)
    if isinstance(name, str):
        return name
    if isinstance(tool, dict):
        value = tool.get("name")
        if isinstance(value, str):
            return value
    return None


def prepare_forked_subagent_spec(
    spec: SubAgent,
    *,
    parent_system_prompt: str | SystemMessage,
    sync_prompt_padding_text: str,
    async_prompt_padding_text: str | None,
    middleware_before_sync_padding: Sequence[AgentMiddleware[Any, Any, Any]],
    middleware_before_async_padding: Sequence[AgentMiddleware[Any, Any, Any]],
    parent_middleware: Sequence[AgentMiddleware[Any, Any, Any]],
    user_middleware: Sequence[AgentMiddleware[Any, Any, Any]],
    middleware_after_user: Sequence[AgentMiddleware[Any, Any, Any]],
) -> SubAgent:
    """Finalize a forked `SubAgent` spec without leaking recursive task tools."""
    fork_middleware: list[AgentMiddleware[Any, Any, Any]] = [
        *middleware_before_sync_padding,
        _SystemPromptPaddingMiddleware(
            text=sync_prompt_padding_text,
            name="ForkSubAgentSystemPromptPadding",
        ),
        *middleware_before_async_padding,
    ]
    if async_prompt_padding_text is not None:
        fork_middleware.append(
            _SystemPromptPaddingMiddleware(
                text=async_prompt_padding_text,
                name="ForkAsyncSubAgentSystemPromptPadding",
            )
        )
    fork_middleware.extend(parent_middleware)
    fork_middleware.extend(user_middleware)
    fork_middleware.extend(middleware_after_user)

    tool_names = [name for name in (_tool_name(tool) for tool in spec.get("tools", []) or []) if name]
    tools_line = ", ".join(f"`{name}`" for name in tool_names) if tool_names else "(none - rely on built-in filesystem/todo tools)"
    subagent_system_prompt = (
        f"You are running as a forked subagent named `{spec['name']}`. "
        "The system prompt above was inherited verbatim from the parent agent to "
        "preserve prompt-cache reuse; it may mention capabilities that do not apply "
        "to you. Your actual environment:\n"
        f"\n- Your declared tools: {tools_line}"
        "\n- You do NOT have the `task` tool. You cannot spawn further subagents."
        "\n\nYour role as this subagent:\n"
        f"{spec['system_prompt']}"
    )

    return {
        **spec,
        "system_prompt": _system_message_text(parent_system_prompt),
        "middleware": fork_middleware,
        "subagent_system_prompt": subagent_system_prompt,
    }


def create_sub_agent(
    spec: SubAgent,
    *,
    state_schema: type | None = None,
    response_format: ResponseFormat[Any] | type | dict[str, Any] | None = None,
) -> Runnable:
    """Create a runnable agent from a raw `SubAgent` spec.

    This is the shared entrypoint for the `create_agent` path used by
    raw subagent specs. Pre-compiled `CompiledSubAgent` runnables are already
    created by the caller and are handled separately by `SubAgentMiddleware`.

    Args:
        spec: Subagent spec to compile. Must specify `model` and `tools`.
        state_schema: Base graph state schema forwarded to `create_agent` for
            the subagent.
        response_format: Optional response format override for this compiled
            subagent instance.

    Returns:
        Runnable agent ready for task-tool invocation.

    Raises:
        ValueError: If `spec` is missing `model` or `tools`.
    """
    if "model" not in spec:
        msg = f"SubAgent '{spec['name']}' must specify 'model'"
        raise ValueError(msg)
    if "tools" not in spec:
        msg = f"SubAgent '{spec['name']}' must specify 'tools'"
        raise ValueError(msg)

    from deepagents._models import resolve_model  # noqa: PLC0415

    model = resolve_model(spec["model"])
    middleware: list[AgentMiddleware] = list(spec.get("middleware", []))

    interrupt_on = spec.get("interrupt_on")
    if interrupt_on:
        middleware.append(HumanInTheLoopMiddleware(interrupt_on=interrupt_on))

    selected_response_format = response_format if response_format is not None else spec.get("response_format")
    create_agent_kwargs: dict[str, Any] = {
        "system_prompt": spec["system_prompt"],
        "tools": spec["tools"],
        "middleware": middleware,
        "name": spec["name"],
        "response_format": selected_response_format,
    }
    if state_schema is not None:
        create_agent_kwargs["state_schema"] = state_schema

    return create_agent(model, **create_agent_kwargs)


def _get_subagent_response_format(
    runtime: ToolRuntime,
) -> ResponseFormat[Any] | type | dict[str, Any] | None:
    """Return the response format carried in this task tool call's config."""
    config = runtime.config
    configurable = config.get("configurable") if isinstance(config, dict) else None
    if not isinstance(configurable, dict):
        return None
    value = configurable.get(SUBAGENT_RESPONSE_FORMAT_CONFIG_KEY)
    if value is None:
        return None
    return value


def _messages_before_current_task_call(messages: Sequence[Any], tool_call_id: str | None) -> list[Any]:
    """Return parent messages before the AI turn that requested this task call."""
    copied = list(messages)
    if not tool_call_id:
        return copied

    for idx in range(len(copied) - 1, -1, -1):
        msg = copied[idx]
        if isinstance(msg, AIMessage):
            tool_calls = getattr(msg, "tool_calls", None) or []
            if any(call.get("id") == tool_call_id for call in tool_calls):
                return copied[:idx]

    for idx in range(len(copied) - 1, -1, -1):
        msg = copied[idx]
        if isinstance(msg, ToolMessage) and msg.tool_call_id == tool_call_id:
            return copied[:idx]
    return copied


def _subagent_config(runtime: ToolRuntime, *, agent_type: str, is_fork: bool) -> RunnableConfig:
    """Build per-invocation config for a subagent run."""
    configurable: dict[str, Any] = {"ls_agent_type": agent_type}
    if is_fork:
        runtime_configurable = runtime.config.get("configurable") if isinstance(runtime.config, dict) else None
        parent_thread_id = runtime_configurable.get("thread_id") if isinstance(runtime_configurable, dict) else None
        prefix = f"{parent_thread_id}:subagent" if parent_thread_id is not None else "subagent"
        configurable["thread_id"] = f"{prefix}:{runtime.tool_call_id}"

    return {"configurable": configurable}


def _build_task_tool(  # noqa: C901, PLR0915
    subagents: Sequence[SubAgent | CompiledSubAgent],
    task_description: str | None = None,
    *,
    private_state_keys: frozenset[str] = frozenset(),
    state_schema: type | None = None,
) -> BaseTool:
    """Create a task tool from subagent specs.

    Args:
        subagents: List of raw or compiled subagent specs.
        task_description: Custom description for the task tool. If `None`,
            uses default template. Supports `{available_agents}` placeholder.
        private_state_keys: State keys marked with `PrivateStateAttr` that
            should be stripped from parent state before invoking subagents.
        state_schema: Base graph state schema forwarded to raw subagent specs.

    Returns:
        A StructuredTool that can invoke subagents by type.
    """

    def _compile_spec(
        spec: SubAgent | CompiledSubAgent,
        *,
        response_format: ResponseFormat[Any] | type | dict[str, Any] | None = None,
    ) -> CompiledSubAgent:
        """Compile one raw spec or configure one provided runnable."""
        if "runnable" in spec:
            if response_format is not None:
                msg = f'response_schema cannot be used with compiled subagent "{spec["name"]}"; dynamic schemas require a raw SubAgent spec.'
                raise ValueError(msg)

            # Use with_config (not attribute mutation) so the original runnable is
            # untouched and a shared instance can be registered under multiple names.
            compiled = cast("CompiledSubAgent", spec)
            runnable = compiled["runnable"].with_config(
                {
                    "metadata": {"lc_agent_name": spec["name"]},
                    "run_name": spec["name"],
                }
            )
            return {
                "name": spec["name"],
                "description": spec["description"],
                "runnable": runnable,
                "fork": bool(spec.get("fork", False)),
                "subagent_system_prompt": spec.get("subagent_system_prompt", "") or "",
            }
        return {
            "name": spec["name"],
            "description": spec["description"],
            "runnable": create_sub_agent(
                spec,
                state_schema=state_schema,
                response_format=response_format,
            ),
            "fork": bool(spec.get("fork", False)),
            "subagent_system_prompt": spec.get("subagent_system_prompt", "") or "",
        }

    compiled_subagents = [_compile_spec(spec) for spec in subagents]
    subagents_by_name = {spec["name"]: spec for spec in subagents}

    # Build the graphs dict and descriptions from the unified spec list
    subagent_graphs: dict[str, Runnable] = {spec["name"]: spec["runnable"] for spec in compiled_subagents}
    subagent_fork_flags: dict[str, bool] = {spec["name"]: bool(spec.get("fork", False)) for spec in compiled_subagents}
    subagent_system_prompts: dict[str, str] = {spec["name"]: spec.get("subagent_system_prompt", "") or "" for spec in compiled_subagents}
    all_forked = bool(compiled_subagents) and all(subagent_fork_flags.values())

    def _format_agent_line(spec: CompiledSubAgent) -> str:
        if spec.get("fork") and not all_forked:
            return f"- {spec['name']} {FORKED_SUBAGENT_MARKER}: {spec['description']}"
        return f"- {spec['name']}: {spec['description']}"

    subagent_description_str = "\n".join(_format_agent_line(spec) for spec in compiled_subagents)

    # Use custom description if provided, otherwise use default template
    if task_description is None:
        description = TASK_TOOL_DESCRIPTION.format(available_agents=subagent_description_str)
    elif "{available_agents}" in task_description:
        description = task_description.format(available_agents=subagent_description_str)
    else:
        description = task_description

    if all_forked:
        description = description + ALL_FORKED_USAGE_GUIDANCE
    elif any(subagent_fork_flags.values()):
        description = description + FORK_USAGE_GUIDANCE

    def _return_command_with_state_update(result: dict, tool_call_id: str) -> Command:
        # Validate that the result contains a 'messages' key
        if "messages" not in result:
            error_msg = (
                "CompiledSubAgent must return a state containing a 'messages' key. "
                "Custom StateGraphs used with CompiledSubAgent should include 'messages' "
                "in their state schema to communicate results back to the main agent."
            )
            raise ValueError(error_msg)

        state_update = {k: v for k, v in result.items() if k not in _EXCLUDED_STATE_KEYS}

        structured = result.get("structured_response")
        if structured is not None:
            if hasattr(structured, "model_dump_json"):
                content: str = structured.model_dump_json()
            elif dataclasses.is_dataclass(structured) and not isinstance(structured, type):
                content = json.dumps(dataclasses.asdict(structured))
            else:
                content = json.dumps(structured)
        else:
            # Walk back to the last AIMessage with non-empty text. Anthropic
            # occasionally emits a trailing empty `end_turn` AIMessage after a
            # successful final tool call, which would otherwise be forwarded
            # as an empty ToolMessage.
            content = ""
            for msg in reversed(result["messages"]):
                if isinstance(msg, AIMessage):
                    text = msg.text.rstrip() if msg.text else ""
                    if text:
                        content = text
                        break

        return Command(
            update={
                **state_update,
                "messages": [ToolMessage(content, tool_call_id=tool_call_id)],
            }
        )

    def _select_subagent(
        subagent_type: str,
        runtime: ToolRuntime,
    ) -> Runnable:
        """Return the runnable to use for this task invocation."""
        response_format = _get_subagent_response_format(runtime)
        if response_format is not None:
            new_spec = _compile_spec(
                subagents_by_name[subagent_type],
                response_format=response_format,
            )
            return new_spec["runnable"]

        return subagent_graphs[subagent_type]

    def _validate_and_prepare_state(
        subagent_type: str,
        description: str,
        runtime: ToolRuntime,
    ) -> tuple[Runnable, dict, bool]:
        """Prepare state for invocation."""
        subagent = _select_subagent(subagent_type, runtime)
        is_fork = subagent_fork_flags.get(subagent_type, False)
        if is_fork:
            subagent_state = {k: runtime.state[k] for k in _FORK_INHERITED_STATE_KEYS if k in runtime.state}
            parent_messages = _messages_before_current_task_call(
                runtime.state.get("messages", []),
                runtime.tool_call_id,
            )
            preamble = subagent_system_prompts.get(subagent_type, "")
            final_content = f"{preamble}\n\n{description}" if preamble else description
            subagent_state["messages"] = [*parent_messages, HumanMessage(content=final_content)]
        else:
            # Create a new state dict to avoid mutating the original
            subagent_state = {k: v for k, v in runtime.state.items() if k not in _EXCLUDED_STATE_KEYS}
            subagent_state = {k: v for k, v in subagent_state.items() if k not in private_state_keys}
            subagent_state["messages"] = [HumanMessage(content=description)]
        return subagent, subagent_state, is_fork

    def task(
        description: str,
        subagent_type: str,
        runtime: ToolRuntime,
    ) -> str | Command:
        if subagent_type not in subagent_graphs:
            allowed_types = ", ".join([f"`{k}`" for k in subagent_graphs])
            return f"We cannot invoke subagent {subagent_type} because it does not exist, the only allowed types are {allowed_types}"
        if not runtime.tool_call_id:
            value_error_msg = "Tool call ID is required for subagent invocation"
            raise ValueError(value_error_msg)
        subagent, subagent_state, is_fork = _validate_and_prepare_state(
            subagent_type,
            description,
            runtime,
        )
        # The parent's callbacks, tags and configurable reach the subagent
        # automatically: langgraph's `ensure_config` seeds each run from the
        # ambient parent config and (as of langgraph#7926) merges it per-key, so
        # the subagent's bound config still wins collisions (e.g. `lc_agent_name`,
        # `recursion_limit`) and parent metadata propagates (deepagents#3634).
        # Forwarding those keys explicitly would double-count under the merge
        # (e.g. duplicate `tags`), so we only stamp the subagent tracing tag.
        agent_type = "fork-subagent" if is_fork else "subagent"
        subagent_config = _subagent_config(runtime, agent_type=agent_type, is_fork=is_fork)
        with _subagent_tracing_context(agent_type):
            result = subagent.invoke(subagent_state, subagent_config)
        return _return_command_with_state_update(result, runtime.tool_call_id)

    async def atask(
        description: str,
        subagent_type: str,
        runtime: ToolRuntime,
    ) -> str | Command:
        if subagent_type not in subagent_graphs:
            allowed_types = ", ".join([f"`{k}`" for k in subagent_graphs])
            return f"We cannot invoke subagent {subagent_type} because it does not exist, the only allowed types are {allowed_types}"
        if not runtime.tool_call_id:
            value_error_msg = "Tool call ID is required for subagent invocation"
            raise ValueError(value_error_msg)
        subagent, subagent_state, is_fork = _validate_and_prepare_state(
            subagent_type,
            description,
            runtime,
        )
        # The parent's callbacks, tags and configurable reach the subagent
        # automatically: langgraph's `ensure_config` seeds each run from the
        # ambient parent config and (as of langgraph#7926) merges it per-key, so
        # the subagent's bound config still wins collisions (e.g. `lc_agent_name`,
        # `recursion_limit`) and parent metadata propagates (deepagents#3634).
        # Forwarding those keys explicitly would double-count under the merge
        # (e.g. duplicate `tags`), so we only stamp the subagent tracing tag.
        agent_type = "fork-subagent" if is_fork else "subagent"
        subagent_config = _subagent_config(runtime, agent_type=agent_type, is_fork=is_fork)
        with _subagent_tracing_context(agent_type):
            result = await subagent.ainvoke(subagent_state, subagent_config)
        return _return_command_with_state_update(result, runtime.tool_call_id)

    return StructuredTool.from_function(
        name="task",
        func=task,
        coroutine=atask,
        description=description,
        infer_schema=False,
        args_schema=TaskToolSchema,
    )


class SubAgentMiddleware(AgentMiddleware[Any, ContextT, ResponseT]):
    """Middleware for providing subagents to an agent via a `task` tool.

    This middleware adds a `task` tool to the agent that can be used
    to invoke subagents.

    Subagents are useful for handling complex tasks that require multiple steps,
    or tasks that require a lot of context to resolve.

    A chief benefit of subagents is that they can handle multi-step tasks,
    and then return a clean, concise response to the main agent.

    Subagents are also great for different domains of expertise that require
    a narrower subset of tools and focus.

    Args:
        backend: Backend for file operations and execution.
        subagents: List of fully-specified subagent configs.

            Each SubAgent must specify `model` and `tools`.

            Optional `interrupt_on` on individual subagents is respected.
        system_prompt: Instructions appended to main agent's system prompt
            about how to use the task tool.
        task_description: Custom description for the task tool.
        state_schema: Base graph state schema forwarded to raw `SubAgent`
            specs when their runnables are compiled.

            Leave unset to use `create_agent`'s default. `CompiledSubAgent`
            entries are unaffected — callers own those runnables' schemas.

    Example:
        ```python
        from deepagents.middleware import SubAgentMiddleware
        from langchain.agents import create_agent

        agent = create_agent(
            "openai:gpt-5.5",
            middleware=[
                SubAgentMiddleware(
                    backend=my_backend,
                    subagents=[
                        {
                            "name": "researcher",
                            "description": "Research agent",
                            "system_prompt": "You are a researcher.",
                            "model": "openai:gpt-5.5",
                            "tools": [search_tool],
                        }
                    ],
                )
            ],
        )
        ```

    """

    def __init__(
        self,
        *,
        backend: BackendProtocol | BackendFactory,
        subagents: Sequence[SubAgent | CompiledSubAgent],
        system_prompt: str | None = TASK_SYSTEM_PROMPT,
        task_description: str | None = None,
        private_state_keys: frozenset[str] | None = None,
        state_schema: type | None = None,
    ) -> None:
        """Initialize the `SubAgentMiddleware`."""
        super().__init__()

        if not subagents:
            msg = "At least one subagent must be specified"
            raise ValueError(msg)
        for spec in subagents:
            if "runnable" in spec and spec.get("fork"):
                msg = f"CompiledSubAgent '{spec['name']}' cannot set fork=True. Compiled subagents own their system prompt and graph."
                raise ValueError(msg)
        self._backend = backend
        self._subagents = subagents
        self._private_state_keys = private_state_keys or frozenset()
        self._task_description = task_description
        self._state_schema = state_schema
        self.subagent_names: frozenset[str] = frozenset(spec["name"] for spec in subagents)
        """Declared subagent names. Public so streamers can discover them
        without introspecting the `task` tool's closure."""

        task_tool = _build_task_tool(
            self._subagents,
            task_description,
            private_state_keys=self._private_state_keys,
            state_schema=self._state_schema,
        )

        self.system_prompt = _build_subagent_system_prompt(subagents, system_prompt)

        self.tools = [task_tool]

    @property
    def private_state_keys(self) -> frozenset[str]:
        """State keys stripped from parent state before invoking subagents."""
        return self._private_state_keys

    @private_state_keys.setter
    def private_state_keys(self, value: frozenset[str]) -> None:
        self._private_state_keys = value
        task_tool = _build_task_tool(
            self._subagents,
            task_description=self._task_description,
            private_state_keys=value,
            state_schema=self._state_schema,
        )
        self.tools = [task_tool]

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        """Update the system message to include instructions on using subagents."""
        if self.system_prompt is not None:
            new_system_message = append_to_system_message(request.system_message, self.system_prompt)
            return handler(request.override(system_message=new_system_message))
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT]:
        """(async) Update the system message to include instructions on using subagents."""
        if self.system_prompt is not None:
            new_system_message = append_to_system_message(request.system_message, self.system_prompt)
            return await handler(request.override(system_message=new_system_message))
        return await handler(request)
