"""An agent definition: what `create_agent` assembles, as data instead of a graph.

The model call, structured-output handling, and the composition of
`wrap_model_call` and `wrap_tool_call` middleware follow
`langchain.agents.create_agent`, reusing its helpers. `PatchToolCallsMiddleware`
is dropped: its rewrite of dangling tool calls is what model-context
derivation already does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware
from langchain.agents.factory import (
    DYNAMIC_TOOL_ERROR_TEMPLATE,
    _chain_async_model_call_handlers,
    _chain_async_tool_call_wrappers,
    _chain_tool_call_wrappers,
    _handle_structured_output_error,
    _is_openai_compatible_model,
    _supports_provider_strategy,
)
from langchain.agents.middleware.types import AgentMiddleware, AgentState, ModelRequest, ModelResponse
from langchain.agents.structured_output import (
    AutoStrategy,
    MultipleStructuredOutputsError,
    OutputToolBinding,
    ProviderStrategy,
    ProviderStrategyBinding,
    StructuredOutputValidationError,
    ToolStrategy,
)
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt.tool_node import ToolNode

from deepagents_durable.schema import Schema

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from langchain_core.language_models import BaseChatModel
    from langchain_core.runnables import Runnable, RunnableConfig
    from langgraph.store.base import BaseStore


def _overrides(middleware: AgentMiddleware, hook: str) -> bool:
    """Whether a middleware implements a hook, sync or async."""
    cls = type(middleware)
    return getattr(cls, hook) is not getattr(AgentMiddleware, hook) or getattr(cls, f"a{hook}") is not getattr(AgentMiddleware, f"a{hook}")


@dataclass
class AgentSpec:
    """Everything a run needs to execute one agent's loop."""

    name: str | None
    model: BaseChatModel
    system_message: SystemMessage | None
    middleware: list[AgentMiddleware]
    tool_node: ToolNode | None
    wraps_tools: bool
    """Whether middleware wraps tool calls, and so may execute tools added at runtime."""
    default_tools: list[BaseTool | dict[str, Any]]
    response_format: Any
    tool_strategy: ToolStrategy[Any] | None
    output_tools: dict[str, OutputToolBinding[Any]]
    schema: Schema
    context_schema: type | None
    store: BaseStore | None
    config: RunnableConfig
    before_agent: list[AgentMiddleware]
    before_model: list[AgentMiddleware]
    after_model: list[AgentMiddleware]
    after_agent: list[AgentMiddleware]
    wrap_model_call: Callable[..., Any] | None

    def tool(self, name: str) -> BaseTool | None:
        """A client-side tool by name."""
        return None if self.tool_node is None else self.tool_node.tools_by_name.get(name)


def _response_format(response_format: Any) -> tuple[Any, ToolStrategy[Any] | None, dict[str, OutputToolBinding[Any]]]:  # noqa: ANN401  # any response format
    if response_format is None:
        initial = None
    elif isinstance(response_format, (ToolStrategy, ProviderStrategy, AutoStrategy)):
        initial = response_format
    else:
        initial = AutoStrategy(schema=response_format)
    setup = ToolStrategy(schema=initial.schema) if isinstance(initial, AutoStrategy) else initial if isinstance(initial, ToolStrategy) else None
    output_tools: dict[str, OutputToolBinding[Any]] = {}
    for spec in setup.schema_specs if setup else []:
        binding = OutputToolBinding.from_schema_spec(spec)
        output_tools[binding.tool.name] = binding
    return initial, setup, output_tools


def build(  # mirrors create_agent's parameters
    model: str | BaseChatModel,
    tools: Sequence[BaseTool | Callable[..., Any] | dict[str, Any]] | None,
    *,
    system_prompt: str | SystemMessage | None,
    middleware: Sequence[AgentMiddleware[Any, Any, Any]],
    response_format: Any,  # noqa: ANN401  # any response format
    state_schema: type | None,
    context_schema: type | None,
    store: BaseStore | None,
    name: str | None,
) -> AgentSpec:
    """Assemble an agent the way `create_agent` does."""
    if isinstance(model, str):
        model = init_chat_model(model)
    system_message = SystemMessage(content=system_prompt) if isinstance(system_prompt, str) else system_prompt
    middleware = [m for m in middleware if not isinstance(m, PatchToolCallsMiddleware)]
    if len({m.name for m in middleware}) != len(middleware):
        msg = "Please remove duplicate middleware instances."
        raise AssertionError(msg)
    initial, setup, output_tools = _response_format(response_format)
    tool_wrappers = [m for m in middleware if _overrides(m, "wrap_tool_call")]
    tools = list(tools or [])
    client_tools = [*(t for m in middleware for t in getattr(m, "tools", [])), *(t for t in tools if not isinstance(t, dict))]
    tool_node = None
    if client_tools or tool_wrappers:
        tool_node = ToolNode(
            tools=client_tools,
            wrap_tool_call=_chain_tool_call_wrappers([m.wrap_tool_call for m in tool_wrappers]),
            awrap_tool_call=_chain_async_tool_call_wrappers([m.awrap_tool_call for m in tool_wrappers]),
        )
    built_in: list[BaseTool | dict[str, Any]] = [t for t in tools if isinstance(t, dict)]
    default_tools: list[BaseTool | dict[str, Any]] = [*tool_node.tools_by_name.values(), *built_in] if tool_node else built_in
    model_wrappers = [m.awrap_model_call for m in middleware if _overrides(m, "wrap_model_call")]
    schema = Schema([*(m.state_schema for m in middleware), state_schema or AgentState])
    config: RunnableConfig = {"recursion_limit": 9_999, "metadata": {"ls_integration": "langchain_create_agent"}}
    if name:
        config["metadata"]["lc_agent_name"] = name
    return AgentSpec(
        name=name,
        model=model,
        system_message=system_message,
        middleware=middleware,
        tool_node=tool_node,
        wraps_tools=bool(tool_wrappers),
        default_tools=default_tools,
        response_format=initial,
        tool_strategy=setup,
        output_tools=output_tools,
        schema=schema,
        context_schema=context_schema,
        store=store,
        config=config,
        before_agent=[m for m in middleware if _overrides(m, "before_agent")],
        before_model=[m for m in middleware if _overrides(m, "before_model")],
        after_model=[m for m in middleware if _overrides(m, "after_model")],
        after_agent=[m for m in middleware if _overrides(m, "after_agent")],
        wrap_model_call=_chain_async_model_call_handlers(model_wrappers),
    )


def _effective_format(spec: AgentSpec, request: ModelRequest) -> Any:  # noqa: ANN401  # any response format
    response_format = request.response_format
    if response_format is not None and not isinstance(response_format, (AutoStrategy, ToolStrategy, ProviderStrategy)):
        response_format = AutoStrategy(schema=response_format)
    if not isinstance(response_format, AutoStrategy):
        return response_format
    if _supports_provider_strategy(request.model, tools=request.tools):
        return ProviderStrategy(schema=response_format.schema)
    if response_format is spec.response_format and spec.tool_strategy is not None:
        return spec.tool_strategy
    return ToolStrategy(schema=response_format.schema)


def bind_model(spec: AgentSpec, request: ModelRequest) -> tuple[Runnable[Any, Any], Any]:
    """The model bound to the request's tools, and the effective response format."""
    if not spec.wraps_tools:
        known = spec.tool_node.tools_by_name if spec.tool_node else {}
        unknown = [t.name for t in request.tools if isinstance(t, BaseTool) and t.name not in known]
        if unknown:
            raise ValueError(DYNAMIC_TOOL_ERROR_TEMPLATE.format(unknown_tool_names=unknown, available_tool_names=sorted(known)))
    effective = _effective_format(spec, request)
    tools = list(request.tools)
    if isinstance(effective, ToolStrategy):
        tools.extend(binding.tool for binding in spec.output_tools.values())
    if isinstance(effective, ProviderStrategy):
        kwargs: dict[str, Any] = {**effective.to_model_kwargs(), **request.model_settings}
        if _is_openai_compatible_model(request.model) and not getattr(request.model, "use_responses_api", False):
            kwargs["strict"] = True
        return request.model.bind_tools(tools, **kwargs), effective
    if isinstance(effective, ToolStrategy):
        for schema_spec in effective.schema_specs:
            if schema_spec.name not in spec.output_tools:
                msg = (
                    f"ToolStrategy specifies tool '{schema_spec.name}' which wasn't declared in the original response format when creating the agent."
                )
                raise ValueError(msg)
        tool_choice = "any" if spec.output_tools else request.tool_choice
        return request.model.bind_tools(tools, tool_choice=tool_choice, **request.model_settings), effective
    if tools:
        return request.model.bind_tools(tools, tool_choice=request.tool_choice, **request.model_settings), None
    return request.model.bind(**request.model_settings), None


def _structured_tool_output(spec: AgentSpec, output: AIMessage, effective: ToolStrategy[Any]) -> dict[str, Any] | None:
    calls = [call for call in output.tool_calls if call["name"] in spec.output_tools]
    if not calls:
        return None
    if len(calls) > 1:
        error = MultipleStructuredOutputsError([call["name"] for call in calls], output)
        retry, message = _handle_structured_output_error(error, effective)
        if not retry:
            raise error
        return {"messages": [output, *(ToolMessage(content=message, tool_call_id=call["id"], name=call["name"]) for call in calls)]}
    call = calls[0]
    try:
        structured = spec.output_tools[call["name"]].parse(call["args"])
    except Exception as exc:
        error = StructuredOutputValidationError(call["name"], exc, output)
        retry, message = _handle_structured_output_error(error, effective)
        if not retry:
            raise error from exc
        return {"messages": [output, ToolMessage(content=message, tool_call_id=call["id"], name=call["name"])]}
    content = effective.tool_message_content or f"Returning structured response: {structured}"
    tool_message = ToolMessage(content=content, tool_call_id=call["id"], name=call["name"])
    return {"messages": [output, tool_message], "structured_response": structured}


def handle_output(spec: AgentSpec, output: AIMessage, effective: Any) -> dict[str, Any]:  # noqa: ANN401  # any response format
    """The model node's update for one model output, including structured responses."""
    if isinstance(effective, ProviderStrategy):
        if output.tool_calls:
            return {"messages": [output]}
        binding = ProviderStrategyBinding.from_schema_spec(effective.schema_spec)
        try:
            return {"messages": [output], "structured_response": binding.parse(output)}
        except Exception as exc:
            name = getattr(effective.schema_spec.schema, "__name__", "response_format")
            raise StructuredOutputValidationError(name, exc, output) from exc
    if isinstance(effective, ToolStrategy) and output.tool_calls:
        handled = _structured_tool_output(spec, output, effective)
        if handled is not None:
            return handled
    return {"messages": [output]}


async def execute_model(spec: AgentSpec, request: ModelRequest) -> ModelResponse:
    """The innermost model call `wrap_model_call` middleware wraps."""
    model, effective = bind_model(spec, request)
    messages = [request.system_message, *request.messages] if request.system_message else request.messages
    output = await model.ainvoke(messages)
    if spec.name:
        output.name = spec.name
    handled = handle_output(spec, output, effective)
    return ModelResponse(result=handled["messages"], structured_response=handled.get("structured_response"))
