# Deep Agents SDK Tests

## API Keys

### Required

- **`ANTHROPIC_API_KEY`** - Required for integration tests using `ChatAnthropic`

### Optional

- **`LANGSMITH_API_KEY`** or **`LANGCHAIN_API_KEY`** - Enables LangSmith tracing for test runs
- **`DEEPAGENTS_RUN_LIVE_PROMPT_CACHE_TESTS=1`** with **`OPENAI_API_KEY`** - Enables opt-in live prompt-cache verification tests
- **`DEEPAGENTS_OPENAI_PROMPT_CACHE_MODEL`** - Overrides the OpenAI model used by the live prompt-cache verification tests

## Test Utilities

Shared test utilities are in `utils.py`:

- Mock tools (`get_weather`, `get_soccer_scores`, etc.)
- Middleware classes (`ResearchMiddleware`, `WeatherToolMiddleware`, etc.)
- Assertion helpers (`assert_all_deepagent_qualities`)
