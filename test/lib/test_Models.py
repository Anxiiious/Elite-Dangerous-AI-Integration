from pathlib import Path
import sys
from unittest.mock import MagicMock
import pytest

from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from src.plugins.MistralPlugin import MistralLLMModel
from src.lib.Models import OpenAILLMModel, OpenAIResponsesLLMModel


class ContentChunk:
    def __init__(self, data: dict):
        self.data = data

    def model_dump(self) -> dict:
        return self.data


class RawResponse:
    def __init__(self, value, retries_taken: int = 0):
        self.value = value
        self.retries_taken = retries_taken

    def parse(self):
        return self.value


def create_chat_completion(text: str = "Recovered") -> ChatCompletion:
    message = ChatCompletionMessage.model_construct(role="assistant", content=text)
    return ChatCompletion.model_construct(
        id="completion-retry",
        choices=[Choice.model_construct(index=0, message=message, finish_reason="stop")],
        created=0,
        model="test-model",
        object="chat.completion",
        usage=None,
    )


def test_chat_completion_reports_sdk_retry_attempts() -> None:
    model = OpenAILLMModel(
        base_url="https://api.mistral.ai/v1",
        api_key="test-key",
        model_name="mistral-test",
        temperature=1.0,
        provider_name="mistral",
    )
    model.client.chat.completions.with_raw_response.create = MagicMock(
        return_value=RawResponse(create_chat_completion(), retries_taken=2)
    )

    text, _, usage = model.generate([{"role": "user", "content": "Hello"}])

    assert text == "Recovered"
    assert usage.retry_attempts == 2
    assert model.client.max_retries == 4


def test_responses_reports_sdk_retry_attempts() -> None:
    model = OpenAIResponsesLLMModel(
        base_url="https://api.openai.com/v1",
        api_key="test-key",
        model_name="gpt-test",
        temperature=1.0,
        provider_name="openai",
    )
    response = MagicMock()
    response.error = None
    response.usage = None
    response.output_text = "Recovered"
    response.output = []
    model.client.responses.with_raw_response.create = MagicMock(
        return_value=RawResponse(response, retries_taken=1)
    )

    text, _, usage = model.generate([{"role": "user", "content": "Hello"}])

    assert text == "Recovered"
    assert usage.retry_attempts == 1
    assert model.client.max_retries == 4


@pytest.mark.parametrize("model_name,configured,expected", [
    ("gpt-6.1-sol", "none", "low"),
    ("gpt-6-astra", "disabled", "low"),
    ("gpt-6.1-sol-2026-10-01", "minimal", "low"),
    ("gpt-6-sol", "none", "none"),
    ("gpt-6-luna", "none", "none"),
    ("gpt-6-luna", "disabled", "none"),
    ("gpt-6-luna", "minimal", "low"),
    ("gpt-6-luna", "default", "medium"),
    ("gpt-6.1-sol", None, "medium"),
    ("gpt-6-astra", "", "medium"),
    ("gpt-6.1-sol", "high", "high"),
    ("gpt-6-astra", "xhigh", "xhigh"),
    ("gpt-6-astra", "max", "max"),
])
def test_gpt6_responses_use_supported_reasoning_and_sampling(model_name, configured, expected) -> None:
    model = OpenAIResponsesLLMModel(
        base_url="https://api.openai.com/v1", api_key="test-key",
        model_name=model_name, temperature=0.7, reasoning_effort=configured,
    )
    response = MagicMock(error=None, usage=None, output_text="Ready", output=[])
    create = MagicMock(return_value=RawResponse(response))
    model.client.responses.with_raw_response.create = create

    model.generate([{"role": "user", "content": "Status?"}])

    request = create.call_args.kwargs
    assert request["reasoning"] == {"effort": expected}
    assert request["text"] == {"verbosity": "low"}
    if expected == "none":
        assert request["temperature"] == 0.7
    else:
        assert "temperature" not in request


def test_gpt6_reasoning_filters_extra_parameters_without_mutating_settings() -> None:
    extra = {
        "temperature": 0.4, "top_p": 0.9, "top_logprobs": 3,
        "reasoning": {"effort": "none", "summary": "auto"},
        "include": ["message.output_text.logprobs", "reasoning.encrypted_content"],
    }
    model = OpenAIResponsesLLMModel(
        base_url="https://api.openai.com/v1", api_key="test-key",
        model_name="gpt-6.1-sol", temperature=1.0, extra_body=extra,
    )
    response = MagicMock(error=None, usage=None, output_text="Ready", output=[])
    create = MagicMock(return_value=RawResponse(response))
    model.client.responses.with_raw_response.create = create

    model.generate([{"role": "user", "content": "Status?"}])

    request = create.call_args.kwargs
    assert request["reasoning"] == {"effort": "low", "summary": "auto"}
    assert request["extra_body"] == {"include": ["reasoning.encrypted_content"]}
    assert extra["temperature"] == 0.4
    assert extra["reasoning"]["effort"] == "none"
    assert len(extra["include"]) == 2


@pytest.mark.parametrize("model_name", ["gpt-5.4-mini", "custom-model", "gpt-6-luna-custom"])
def test_gpt6_compatibility_preserves_other_models(model_name) -> None:
    model = OpenAIResponsesLLMModel(
        base_url="https://example.test/v1", api_key="test-key",
        model_name=model_name, temperature=0.7, reasoning_effort="low",
    )
    response = MagicMock(error=None, usage=None, output_text="Ready", output=[])
    create = MagicMock(return_value=RawResponse(response))
    model.client.responses.with_raw_response.create = create

    model.generate([{"role": "user", "content": "Status?"}])

    request = create.call_args.kwargs
    assert request["temperature"] == 0.7
    assert request["reasoning"] == {"effort": "low"}


def test_gpt6_sdk_serializes_vision_and_tool_round_trip() -> None:
    import json
    import httpx

    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        output = ([{
            "type": "function_call", "id": "fc_1", "call_id": "call_1",
            "name": "get_status", "arguments": "{}", "status": "completed",
        }] if len(requests) == 1 else [{
            "type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": "Shields online.", "annotations": []}],
        }])
        return httpx.Response(200, json={
            "id": "resp_1", "object": "response", "created_at": 0,
            "model": "gpt-6.1-sol", "status": "completed", "error": None,
            "output": output,
        })

    from openai import OpenAI
    model = OpenAIResponsesLLMModel(
        base_url="https://api.openai.com/v1", api_key="test-key",
        model_name="gpt-6.1-sol", temperature=1.0, reasoning_effort="none",
    )
    with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
        model.client = OpenAI(api_key="test-key", http_client=http_client)
        messages = [{"role": "user", "content": [
            {"type": "text", "text": "Check shields"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,dGVzdA=="}},
        ]}]
        tools = [{"type": "function", "function": {
            "name": "get_status", "parameters": {"type": "object", "properties": {}},
        }}]
        _, calls, _ = model.generate(messages, tools, "auto")
        assert calls and calls[0].function.name == "get_status"
        messages.extend([
            {"role": "assistant", "content": None, "tool_calls": [calls[0].model_dump()]},
            {"role": "tool", "tool_call_id": calls[0].id, "content": "Shields online"},
        ])
        text, _, _ = model.generate(messages, tools)

    assert text == "Shields online."
    assert requests[0]["input"][0]["content"][1]["type"] == "input_image"
    assert requests[0]["tools"][0]["name"] == "get_status"
    assert requests[1]["input"][-1]["type"] == "function_call_output"
    assert requests[1]["input"][-1]["call_id"] == "call_1"
    assert all(request["reasoning"]["effort"] == "low" for request in requests)
    assert all("temperature" not in request for request in requests)


def test_chat_completion_extracts_visible_text_from_structured_content() -> None:
    model = MistralLLMModel(
        base_url="https://api.mistral.ai/v1",
        api_key="test-key",
        model_name="mistral-medium-latest",
        temperature=1.0,
        provider_name="mistral",
    )
    message = ChatCompletionMessage.model_construct(
        role="assistant",
        content=[
            ContentChunk({"type": "thinking", "thinking": [{"type": "text", "text": "secret"}]}),
            ContentChunk({"type": "text", "text": "Final report"}),
            ContentChunk({"type": "text", "text": " with sources."}),
        ],
    )
    completion = ChatCompletion.model_construct(
        id="completion-1",
        choices=[Choice.model_construct(index=0, message=message, finish_reason="stop")],
        created=0,
        model="mistral-medium-latest",
        object="chat.completion",
        usage=None,
    )
    model.client.chat.completions.with_raw_response.create = MagicMock(
        return_value=RawResponse(completion)
    )

    text, tool_calls, usage = model.generate([{"role": "user", "content": "Search"}])

    assert text == "Final report with sources."
    assert tool_calls is None
    assert usage.output_chars == len(text)


def test_chat_completion_returns_none_for_thinking_only_content() -> None:
    model = MistralLLMModel(
        base_url="https://api.mistral.ai/v1",
        api_key="test-key",
        model_name="mistral-medium-latest",
        temperature=1.0,
        provider_name="mistral",
    )
    message = ChatCompletionMessage.model_construct(
        role="assistant",
        content=[ContentChunk({"type": "thinking", "thinking": []})],
    )
    completion = ChatCompletion.model_construct(
        id="completion-1",
        choices=[Choice.model_construct(index=0, message=message, finish_reason="stop")],
        created=0,
        model="mistral-medium-latest",
        object="chat.completion",
        usage=None,
    )
    model.client.chat.completions.with_raw_response.create = MagicMock(
        return_value=RawResponse(completion)
    )

    text, tool_calls, _ = model.generate([{"role": "user", "content": "Search"}])

    assert text is None
    assert tool_calls is None


def test_chat_completion_strips_thinking_chunks_from_tool_prompt() -> None:
    model = MistralLLMModel(
        base_url="https://api.mistral.ai/v1",
        api_key="test-key",
        model_name="mistral-small-latest",
        temperature=1.0,
        provider_name="mistral",
    )
    message = ChatCompletionMessage.model_construct(role="assistant", content="Acknowledged")
    completion = ChatCompletion.model_construct(
        id="completion-1",
        choices=[Choice.model_construct(index=0, message=message, finish_reason="stop")],
        created=0,
        model="mistral-small-latest",
        object="chat.completion",
        usage=None,
    )
    model.client.chat.completions.with_raw_response.create = MagicMock(
        return_value=RawResponse(completion)
    )
    tool_content = [
        ContentChunk({"type": "thinking", "thinking": [{"type": "text", "text": "secret"}]}),
        ContentChunk({"type": "text", "text": "Visible search result"}),
    ]

    model.generate([
        {"role": "assistant", "content": None, "tool_calls": []},
        {"role": "tool", "tool_call_id": "call-1", "content": tool_content},
    ])

    request_messages = model.client.chat.completions.with_raw_response.create.call_args.kwargs["messages"]
    assert request_messages[1]["content"] == "Visible search result"
    assert request_messages[1]["tool_call_id"] == "call-1"
    assert "secret" not in request_messages[1]["content"]
    assert tool_content[0].data["type"] == "thinking"


def test_chat_completion_json_encodes_structured_tool_prompt() -> None:
    model = MistralLLMModel(
        base_url="https://api.mistral.ai/v1",
        api_key="test-key",
        model_name="mistral-small-latest",
        temperature=1.0,
        provider_name="mistral",
    )
    message = ChatCompletionMessage.model_construct(role="assistant", content="Acknowledged")
    completion = ChatCompletion.model_construct(
        id="completion-1",
        choices=[Choice.model_construct(index=0, message=message, finish_reason="stop")],
        created=0,
        model="mistral-small-latest",
        object="chat.completion",
        usage=None,
    )
    model.client.chat.completions.with_raw_response.create = MagicMock(
        return_value=RawResponse(completion)
    )

    model.generate([
        {"role": "tool", "tool_call_id": "call-1", "content": {"systems": ["Sol"]}},
    ])

    request_messages = model.client.chat.completions.with_raw_response.create.call_args.kwargs["messages"]
    assert request_messages[0]["content"] == '{"systems": ["Sol"]}'
