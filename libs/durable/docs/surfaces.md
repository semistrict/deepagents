# What the durable runtime must provide

Notes from surveying the code that runs on LangGraph today. Paths are relative to `libs/` unless they start with `~`.

## 1. The deepagents middleware stack (must run unchanged)

### Agent loop semantics (mirrors `langchain.agents.create_agent`, `~/src/langchain-python/libs/langchain_v1/langchain/agents/factory.py`)

- Hooks:
  - `before_agent` and `before_model` run in middleware order.
  - `after_model` and `after_agent` run in reverse order.
  - `wrap_model_call` and `wrap_tool_call` compose with the first middleware outermost.
- A hook may declare `config` in its signature and receive it as a keyword argument.
- Sync-only hooks run inline under async.
- Model node:
  - Builds `ModelRequest(model, tools, system_message, response_format, messages, tool_choice=None, state, runtime)` and runs the composed `wrap_model_call` around the model call.
  - Returns `[Command(update={"messages": result, "structured_response"?}), *middleware_commands]`. Middleware commands are inner-first, then outer.
  - All of these writes land in one step and reach the reducers in order.
- `ExtendedModelResponse(model_response, command)` is unwrapped at layer boundaries. A wrap_model_call command may not use `goto`, `resume` or `graph`.
- Routing:
  - `jump_to` is ephemeral and takes `model`, `tools` or `end`.
  - From the model: go to `jump_to`; else exit if the last AI message has no tool calls; else run each pending tool call (one per call, in parallel); else exit if `structured_response` is set; else return to the model (synthetic tool results were injected).
  - After tools: exit if every executed client tool is `return_direct`, or if a structured-output tool ran; otherwise return to the model.
- Structured output (`ToolStrategy`, `ProviderStrategy`, `AutoStrategy`) follows `factory.py` `_handle_model_output` and `_get_bound_model`.
- Tool execution follows `~/src/langgraph/libs/prebuilt/langgraph/prebuilt/tool_node.py`:
  - Injection: `ToolRuntime`, `InjectedState`, `InjectedStore`, `InjectedToolCallId`.
  - Error handling: the default re-raises anything that is not a `ToolInvocationError`. `GraphBubbleUp` always propagates.
  - A tool may return a `Command`; `Command(goto=..., graph=Command.PARENT)` must pass through.

### Pregel internals reached through public helpers

- `get_config()` reads langchain_core's `var_child_runnable_config` and raises `RuntimeError` outside a run (`backends/state.py:57-79` relies on that).
- `configurable["__pregel_runtime"]` backs `get_runtime()`, `get_store()` and `get_stream_writer()`.
- `configurable["__pregel_read"]` / `["__pregel_send"]`:
  - StateBackend does `read("files", True)` and `send([("files", update)])` in every task: tools, the model node (summarization offload, eviction) and `before_agent` (memory, skills).
  - ToolNode requires `read` to be a `functools.partial` whose `.args[1]` is the channel mapping. It then calls `read(list(channels), True)`.
  - `fresh=True` means the task's own writes are applied on top of copied channels; siblings are not visible.
- `interrupt()` (HITL `after_model`) uses `configurable["__pregel_scratchpad"]` and `["__pregel_send"]`.
  - On resume the whole node re-executes, and the Nth `interrupt()` returns the Nth resume value.
  - Payload: `{"action_requests": [...], "review_configs": [...]}`. Resume value: `{"decisions": [...]}`.

### Channels and reducers

- Schemas merge in order: middleware schemas first, then the base schema last, so the base wins.
- `messages`:
  - `DeltaChannel(_messages_delta_reducer)`, a batch reducer over all of a step's writes.
  - The last `RemoveMessage(REMOVE_ALL)` discards prior state.
  - `RemoveMessage(id)` tombstones a message; an existing id replaces in place; a new id appends.
  - IDs are not assigned by the reducer. LangGraph stamps IDs on every write (`ensure_message_ids`).
  - Replace-by-id is load-bearing for HITL edits, eviction, summarization and blob offload.
- `files`: `DeltaChannel(_file_data_delta_reducer)`; a `None` value deletes a key.
- `_blob_payloads`: `UntrackedValue` (never persisted).
- `jump_to`: ephemeral.
- Other fields are last-value, or `Annotated[T, reducer]` (for example `async_tasks` dict-merge).
- `PrivateStateAttr` / `OmitFromSchema`:
  - Input filtering drops private keys on invoke.
  - Output filtering applies to returned state.
  - The `task` tool relies on output filtering.

### Runtime and ToolRuntime fields used

- `Runtime`: `context` (subagents inherit it), `store`, `stream_writer`, and `execution_info.{thread_id, checkpoint_ns, task_id, run_id}` (dcode).
- `ToolRuntime`: `state`, `config` (`recursion_limit`, `tags`, `metadata.lc_agent_name` / `ls_integration`, and `configurable["__deepagents_subagent_response_format"]`), `tool_call_id`, `store`, `context`, `stream_writer`.

### Subagents (`deepagents/middleware/subagents.py`)

- The `task` tool calls `subagent.invoke(state, {"configurable": {"ls_agent_type": "subagent"}})` on graphs built with `create_agent`, which has no checkpointer and inherits the parent's.
- Input state: the parent state minus `_EXCLUDED_STATE_KEYS` and private keys, plus `messages=[HumanMessage(description)]`.
- Output: a `Command(update={...filtered result, "messages": [ToolMessage(last AI text or structured JSON)]})`.
- A subagent interrupt bubbles up to the parent. On resume, the same tool call re-runs.
- Streaming with `subgraphs=True` yields namespaces `("tools:<task_id>",)`. Message metadata carries `lc_agent_name`.

### Test seams

- `tests/unit_tests/test_graph.py` patches `deepagents.graph.create_agent`.
- Fake models live in `tests/unit_tests/chat_model.py` (`GenericFakeChatModel`) and `test_subagents.py` (`_ScriptedChatModel`).
- Parity targets:
  - `test_end_to_end.py` (StateBackend config keys, delta channels, eviction, summarization)
  - `test_subagents.py` (private state, recursion, interrupts, streaming)
  - `test_messages_reducer.py`

## 2. dcode (libs/code)

dcode is deeply coupled to the LangGraph server and its internals:

- **Server:** it runs `langgraph dev` (langgraph-api, which is ELv2-licensed) in a child process. That server hosts the graph factory `deepagents_code.server_graph:make_graph` and the custom routes in `offload_api.py`.
- **Client:** it uses `RemoteGraph` and `langgraph_sdk`:
  - `runs.stream(stream_mode=["messages","updates","custom"], stream_subgraphs=True)`, chunks are `(ns, mode, data)`
  - `threads.get_state` / `update_state(as_node=...)`
  - `runs.list` / `cancel`
  - `store.put_item` (approval mode, read mid-run by middleware)
- **Direct checkpoint SQL:** `sessions.py` (thread listing and message counts from `writes`) and the thread-inspector skill.
- **Pregel internals:** `StreamMessagesHandler` (`model_retry.py`, `cost_tracking.py`), `__pregel_checkpointer` (`_js_cost.py`), the `checkpoint_ns` format.
- **Interrupts:** keyed `Command(resume={interrupt_id: value})` covering hooks, `ask_user` and HITL.

Porting dcode means replacing its launch, streaming and session-listing layers, not swapping a backend.

## 3. useStream (langgraphjs)

- **v1 `@langchain/react`** uses only the new agent streaming protocol (`~/src/agent-protocol/streaming`):
  - `POST /threads/{id}/commands`: `run.start`, `input.respond`.
  - `POST /threads/{id}/stream/events` (SSE) or WebSocket. Events are `{type: "event", event_id, seq, method, params: {namespace, timestamp, node?, data}}` on channels `values`, `updates`, `messages`, `tools`, `lifecycle`, `input`, `checkpoints`, `tasks`, `custom`.
  - The server must replay all of the thread's events on every open. The client deduplicates by `event_id`.
  - It also calls `GET /threads/{id}/state`, `POST /threads/{id}/history` and `POST .../runs/{rid}/cancel`.
- **Legacy `@langchain/langgraph-sdk/react`** uses classic `POST /threads/{id}/runs/stream` SSE with stream modes:
  - Usually `messages-tuple` + `values`, with `values|ns` namespacing.
  - `Content-Location: /threads/{tid}/runs/{rid}` is required.
  - Reconnect is `GET /threads/{id}/runs/{rid}/stream` with `Last-Event-ID`.

Durable commit frames map naturally onto the new protocol's seq-ordered, replayable events.
