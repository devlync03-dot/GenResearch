"""Provider contract and response-safety tests for the Modal LLM client."""
import asyncio
import json

import httpx
import pytest

from config import settings
from services import llm_service
from services.llm_models import ROLE_MODEL_CHAINS, get_chain_for_role


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def _modal_response(content, *, finish_reason="stop"):
    return {
        "choices": [{
            "message": {"content": content},
            "finish_reason": finish_reason,
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


def test_call_llm_uses_only_configured_modal_model_and_caps_output(monkeypatch):
    requests = []

    class FakeClient:
        async def post(self, url, *, json, headers):
            requests.append((url, json, headers))
            return FakeResponse(_modal_response("Modal answer"))

    monkeypatch.setattr(settings, "MODAL_PROXY_TOKEN_ID", "test-id")
    monkeypatch.setattr(settings, "MODAL_PROXY_TOKEN_SECRET", "test-secret")
    monkeypatch.setattr(settings, "MODAL_MAX_TOKENS", 16384)
    monkeypatch.setattr(llm_service, "_get_client", lambda _kind: FakeClient())
    monkeypatch.setattr(llm_service, "_track_in_background", lambda **_kwargs: None)

    answer = asyncio.run(
        llm_service.call_llm(
            "prompt",
            agent_role="literature_review",
            max_tokens=20_000,
        )
    )

    url, payload, headers = requests[0]
    assert answer == "Modal answer"
    assert url == (
        "https://abbasi-kaleem2002--ep-deepseek-v4-1-flash-1017-server."
        "us-west.modal.direct/v1/chat/completions"
    )
    assert payload["model"] == "deepseek-ai/DeepSeek-V4.1-Flash-1017"
    assert payload["max_tokens"] == 16384
    assert payload["top_p"] == 0.9
    assert payload["reasoning_effort"] == "none"
    assert payload["stream"] is False
    assert headers["Authorization"] == "Bearer test-id.test-secret"


@pytest.mark.parametrize("agent_role", ["summarization", "summarization_final"])
def test_summarization_roles_use_lower_configured_reasoning_effort(monkeypatch, agent_role):
    seen = {}

    class FakeClient:
        async def post(self, _url, *, json, headers):
            seen["payload"] = json
            return FakeResponse(_modal_response("Summary answer"))

    monkeypatch.setattr(settings, "MODAL_PROXY_TOKEN_ID", "test-id")
    monkeypatch.setattr(settings, "MODAL_PROXY_TOKEN_SECRET", "test-secret")
    monkeypatch.setattr(settings, "MODAL_REASONING_EFFORT", "high")
    monkeypatch.setattr(settings, "MODAL_SUMMARIZATION_REASONING_EFFORT", "none")
    monkeypatch.setattr(llm_service, "_get_client", lambda _kind: FakeClient())
    monkeypatch.setattr(llm_service, "_track_in_background", lambda **_kwargs: None)

    answer = asyncio.run(
        llm_service.call_llm("summarize this", agent_role=agent_role, max_tokens=3000)
    )

    assert answer == "Summary answer"
    assert seen["payload"]["reasoning_effort"] == "none"
    assert seen["payload"]["max_tokens"] == 3000


def test_literature_review_uses_configured_reasoning_effort(monkeypatch):
    seen = {}

    class FakeClient:
        async def post(self, _url, *, json, headers):
            seen["payload"] = json
            return FakeResponse(_modal_response("Review answer"))

    monkeypatch.setattr(settings, "MODAL_PROXY_TOKEN_ID", "test-id")
    monkeypatch.setattr(settings, "MODAL_PROXY_TOKEN_SECRET", "test-secret")
    monkeypatch.setattr(settings, "MODAL_REASONING_EFFORT", "high")
    monkeypatch.setattr(settings, "MODAL_LITERATURE_REVIEW_REASONING_EFFORT", "none")
    monkeypatch.setattr(llm_service, "_get_client", lambda _kind: FakeClient())
    monkeypatch.setattr(llm_service, "_track_in_background", lambda **_kwargs: None)

    answer = asyncio.run(
        llm_service.call_llm(
            "write a review",
            agent_role="literature_review",
            max_tokens=3200,
        )
    )

    assert answer == "Review answer"
    assert seen["payload"]["reasoning_effort"] == "none"
    assert seen["payload"]["max_tokens"] == 3200


def test_call_llm_honors_explicit_reasoning_effort(monkeypatch):
    seen = {}

    class FakeClient:
        async def post(self, _url, *, json, headers):
            seen["payload"] = json
            return FakeResponse(_modal_response('{"overall_assessment":"needs_more"}'))

    monkeypatch.setattr(settings, "MODAL_PROXY_TOKEN_ID", "test-id")
    monkeypatch.setattr(settings, "MODAL_PROXY_TOKEN_SECRET", "test-secret")
    monkeypatch.setattr(settings, "MODAL_REASONING_EFFORT", "high")
    monkeypatch.setattr(llm_service, "_get_client", lambda _kind: FakeClient())
    monkeypatch.setattr(llm_service, "_track_in_background", lambda **_kwargs: None)

    answer = asyncio.run(
        llm_service.call_llm(
            "evaluate one section",
            agent_role="sufficiency_evaluator",
            reasoning_effort="none",
        )
    )

    assert answer == '{"overall_assessment":"needs_more"}'
    assert seen["payload"]["reasoning_effort"] == "none"


def test_pipeline_roles_default_to_no_reasoning_even_when_global_setting_is_high(monkeypatch):
    seen = []

    class FakeClient:
        async def post(self, _url, *, json, headers):
            seen.append(json)
            return FakeResponse(_modal_response("Generated output"))

    monkeypatch.setattr(settings, "MODAL_PROXY_TOKEN_ID", "test-id")
    monkeypatch.setattr(settings, "MODAL_PROXY_TOKEN_SECRET", "test-secret")
    monkeypatch.setattr(settings, "MODAL_REASONING_EFFORT", "high")
    monkeypatch.setattr(llm_service, "_get_client", lambda _kind: FakeClient())
    monkeypatch.setattr(llm_service, "_track_in_background", lambda **_kwargs: None)

    answer = asyncio.run(
        llm_service.call_llm(
            "draft one section",
            agent_role="draft",
            max_tokens=3000,
        )
    )

    assert answer == "Generated output"
    assert seen[0]["reasoning_effort"] == "none"


def test_call_llm_retries_empty_output_limit_with_larger_budget_and_no_reasoning(monkeypatch):
    requests = []

    class FakeClient:
        async def post(self, _url, *, json, headers):
            requests.append(json)
            if len(requests) == 1:
                return FakeResponse({
                    "choices": [{
                        "message": {"content": ""},
                        "finish_reason": "length",
                    }],
                    "usage": {"completion_tokens": json["max_tokens"]},
                })
            return FakeResponse(_modal_response("Recovered full answer"))

    monkeypatch.setattr(settings, "MODAL_PROXY_TOKEN_ID", "test-id")
    monkeypatch.setattr(settings, "MODAL_PROXY_TOKEN_SECRET", "test-secret")
    monkeypatch.setattr(settings, "MODAL_REASONING_EFFORT", "high")
    monkeypatch.setattr(settings, "MODAL_MAX_TOKENS", 4096)
    monkeypatch.setattr(settings, "MODAL_CONTEXT", 8192)
    monkeypatch.setattr(llm_service, "_get_client", lambda _kind: FakeClient())
    monkeypatch.setattr(llm_service, "_track_in_background", lambda **_kwargs: None)

    answer = asyncio.run(
        llm_service.call_llm(
            "draft one section",
            agent_role="draft",
            max_tokens=1000,
        )
    )

    assert answer == "Recovered full answer"
    assert len(requests) == 2
    assert requests[0]["max_tokens"] == 1000
    assert requests[0]["reasoning_effort"] == "none"
    assert requests[1]["max_tokens"] == 2000
    assert requests[1]["reasoning_effort"] == "none"


def test_json_roles_request_json_object_mode(monkeypatch):
    seen = {}

    class FakeClient:
        async def post(self, _url, *, json, headers):
            seen["payload"] = json
            return FakeResponse(_modal_response('{"ok": true}'))

    monkeypatch.setattr(settings, "MODAL_PROXY_TOKEN_ID", "test-id")
    monkeypatch.setattr(settings, "MODAL_PROXY_TOKEN_SECRET", "test-secret")
    monkeypatch.setattr(llm_service, "_get_client", lambda _kind: FakeClient())
    monkeypatch.setattr(llm_service, "_track_in_background", lambda **_kwargs: None)

    result = asyncio.run(
        llm_service.call_llm("prompt", agent_role="verification")
    )

    assert result == '{"ok": true}'
    assert seen["payload"]["response_format"] == {"type": "json_object"}


def test_modal_empty_response_reports_reasoning_output_limit_without_exposing_reasoning():
    response = {
        "choices": [{
            "message": {
                "content": "",
                "reasoning_content": "private reasoning must not be surfaced",
            },
            "finish_reason": "length",
        }],
        "usage": {"completion_tokens": 1000},
    }

    with pytest.raises(
        llm_service.NonRetryableLLMError,
        match=r"output token limit.*1000 completion tokens.*MODAL_REASONING_EFFORT",
    ) as error:
        llm_service._modal_response(response)

    assert "private reasoning" not in str(error.value)


def test_chat_call_preserves_answer_contract_and_drops_private_reasoning(monkeypatch):
    seen = {}

    class FakeClient:
        async def post(self, _url, *, json, headers):
            seen["payload"] = json
            return FakeResponse(
                _modal_response(
                    "<think>private reasoning</think>"
                    "<summary>Evidence summary.</summary>"
                    "<answer>Grounded answer [Source 1].</answer>"
                )
            )

    monkeypatch.setattr(settings, "MODAL_PROXY_TOKEN_ID", "test-id")
    monkeypatch.setattr(settings, "MODAL_PROXY_TOKEN_SECRET", "test-secret")
    monkeypatch.setattr(llm_service, "_get_client", lambda _kind: FakeClient())

    answer = asyncio.run(llm_service.call_llm("question", agent_role="chat"))

    assert answer == "Grounded answer [Source 1]."
    system_message = seen["payload"]["messages"][0]["content"]
    assert "Do not include private reasoning" in system_message
    assert "private reasoning" not in answer


def test_chat_stream_emits_answer_and_suppresses_private_reasoning(monkeypatch):
    class FakeResponse:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def raise_for_status(self):
            return None

        async def aiter_lines(self):
            for content in (
                "<think>private text</think><summary>Evidence",
                " summary.</summary><answer>Final [Source 1].</answer>",
            ):
                yield "data: " + json.dumps({
                    "choices": [{"delta": {"content": content}, "finish_reason": None}]
                })
            yield "data: [DONE]"

    class FakeClient:
        def stream(self, method, url, *, json, headers):
            assert method == "POST"
            assert url.endswith("/chat/completions")
            assert json["stream"] is True
            return FakeResponse()

    monkeypatch.setattr(settings, "MODAL_PROXY_TOKEN_ID", "test-id")
    monkeypatch.setattr(settings, "MODAL_PROXY_TOKEN_SECRET", "test-secret")
    monkeypatch.setattr(llm_service, "_get_client", lambda _kind: FakeClient())

    async def collect():
        return [
            item
            async for item in llm_service.call_llm_stream(
                "question",
                agent_role="chat",
                emit_events=True,
            )
        ]

    events = asyncio.run(collect())
    answer = "".join(text for event_type, text in events if event_type == "token")
    assert answer == "Final [Source 1]."
    assert all("private text" not in text for _, text in events)
    assert not any(event_type == "thinking" for event_type, _ in events)


def test_chat_stream_warns_when_model_reaches_output_limit(monkeypatch):
    class FakeResponse:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def raise_for_status(self):
            return None

        async def aiter_lines(self):
            yield "data: " + json.dumps({
                "choices": [{
                    "delta": {"content": "Partial answer [Source 1]."},
                    "finish_reason": "length",
                }]
            })
            yield "data: [DONE]"

    class FakeClient:
        def stream(self, *_args, **_kwargs):
            return FakeResponse()

    monkeypatch.setattr(settings, "MODAL_PROXY_TOKEN_ID", "test-id")
    monkeypatch.setattr(settings, "MODAL_PROXY_TOKEN_SECRET", "test-secret")
    monkeypatch.setattr(llm_service, "_get_client", lambda _kind: FakeClient())

    async def collect():
        return [
            item
            async for item in llm_service.call_llm_stream(
                "question",
                agent_role="chat",
                emit_events=True,
            )
        ]

    events = asyncio.run(collect())
    assert events[-1] == (
        "warning",
        "The model reached its output limit. This answer may be incomplete.",
    )


def test_modal_request_retries_transient_connection_failure(monkeypatch):
    calls = []

    class FakeClient:
        async def post(self, url, *, json, headers):
            calls.append(url)
            if len(calls) == 1:
                raise httpx.ConnectError("temporary network failure")
            return FakeResponse(_modal_response("Recovered response."))

    async def no_wait(_seconds):
        return None

    monkeypatch.setattr(settings, "MODAL_PROXY_TOKEN_ID", "test-id")
    monkeypatch.setattr(settings, "MODAL_PROXY_TOKEN_SECRET", "test-secret")
    monkeypatch.setattr(llm_service, "_get_client", lambda _kind: FakeClient())
    monkeypatch.setattr(llm_service.asyncio, "sleep", no_wait)

    answer, usage = asyncio.run(
        llm_service._call_modal_once(
            [{"role": "user", "content": "prompt"}],
            temperature=0,
            max_tokens=50,
            json_mode=False,
        )
    )

    assert answer == "Recovered response."
    assert usage == {"prompt_eval_count": 10, "eval_count": 5}
    assert len(calls) == 2


def test_modal_request_does_not_retry_read_timeout(monkeypatch):
    calls = []

    class FakeClient:
        async def post(self, url, *, json, headers):
            calls.append(url)
            raise httpx.ReadTimeout("model did not produce output")

    monkeypatch.setattr(settings, "MODAL_PROXY_TOKEN_ID", "test-id")
    monkeypatch.setattr(settings, "MODAL_PROXY_TOKEN_SECRET", "test-secret")
    monkeypatch.setattr(llm_service, "_get_client", lambda _kind: FakeClient())

    with pytest.raises(
        llm_service.NonRetryableLLMError,
        match="was not retried",
    ):
        asyncio.run(
            llm_service._call_modal_once(
                [{"role": "user", "content": "prompt"}],
                temperature=0,
                max_tokens=8,
                json_mode=False,
            )
        )

    assert len(calls) == 1


def test_modal_chat_stream_fails_clearly_and_does_not_retry_read_timeout(monkeypatch):
    calls = []

    class FakeResponse:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def raise_for_status(self):
            return None

        async def aiter_lines(self):
            raise httpx.ReadTimeout("model did not produce output")
            yield ""

    class FakeClient:
        def stream(self, *_args, **_kwargs):
            calls.append(1)
            return FakeResponse()

    monkeypatch.setattr(settings, "MODAL_PROXY_TOKEN_ID", "test-id")
    monkeypatch.setattr(settings, "MODAL_PROXY_TOKEN_SECRET", "test-secret")
    monkeypatch.setattr(llm_service, "_get_client", lambda _kind: FakeClient())

    async def collect():
        return [
            item async for item in llm_service._call_modal_chat_stream(
                [{"role": "user", "content": "prompt"}],
                temperature=0,
                max_tokens=8,
            )
        ]

    with pytest.raises(
        llm_service.NonRetryableLLMError,
        match="was not retried",
    ):
        asyncio.run(collect())

    assert len(calls) == 1


def test_missing_modal_key_fails_before_request(monkeypatch):
    monkeypatch.setattr(settings, "MODAL_PROXY_TOKEN_ID", "")
    monkeypatch.setattr(settings, "MODAL_PROXY_TOKEN_SECRET", "")

    with pytest.raises(
        llm_service.NonRetryableLLMError,
        match="MODAL_PROXY_TOKEN_ID and MODAL_PROXY_TOKEN_SECRET",
    ):
        llm_service._modal_headers()


def test_readiness_checks_modal_generation_and_ollama_embedding(monkeypatch):
    class FakeModalClient:
        async def get(self, url, *, headers, timeout):
            assert url.endswith("/models")
            assert headers["Authorization"] == llm_service._modal_headers()["Authorization"]
            return FakeResponse({
                "data": [{"id": "deepseek-ai/DeepSeek-V4.1-Flash-1017"}]
            })

    class FakeOllamaClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, url):
            assert url.endswith("/api/tags")
            return FakeResponse({
                "models": [{"name": "nomic-embed-text:latest"}]
            })

    monkeypatch.setattr(settings, "MODAL_PROXY_TOKEN_ID", "test-id")
    monkeypatch.setattr(settings, "MODAL_PROXY_TOKEN_SECRET", "test-secret")
    monkeypatch.setattr(llm_service, "_get_client", lambda _kind: FakeModalClient())
    monkeypatch.setattr(llm_service.httpx, "AsyncClient", lambda **_kwargs: FakeOllamaClient())

    result = asyncio.run(llm_service.check_provider_readiness())

    assert result["status"] == "ok"
    assert set(result["available_models"]) == {
        "modal:deepseek-ai/DeepSeek-V4.1-Flash-1017",
        "ollama:nomic-embed-text",
    }
    assert result["missing_models"] == []
    assert result["inference_tested"] is False


def test_readiness_reports_missing_modal_key_without_calling_generation_api(monkeypatch):
    class FakeOllamaClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def get(self, _url):
            return FakeResponse({
                "models": [{"name": "nomic-embed-text:latest"}]
            })

    def unexpected_modal_client(_kind):
        raise AssertionError("Modal API should not be called without a key")

    monkeypatch.setattr(settings, "MODAL_PROXY_TOKEN_ID", "")
    monkeypatch.setattr(settings, "MODAL_PROXY_TOKEN_SECRET", "")
    monkeypatch.setattr(llm_service, "_get_client", unexpected_modal_client)
    monkeypatch.setattr(llm_service.httpx, "AsyncClient", lambda **_kwargs: FakeOllamaClient())

    result = asyncio.run(llm_service.check_provider_readiness())

    assert result["status"] == "not_ready"
    assert result["available_models"] == ["ollama:nomic-embed-text"]
    assert result["missing_models"] == ["modal:deepseek-ai/DeepSeek-V4.1-Flash-1017"]
    assert result["errors"] == [
        "MODAL_PROXY_TOKEN_ID and MODAL_PROXY_TOKEN_SECRET must be configured."
    ]


def test_all_generation_roles_use_only_modal_model():
    for role in ROLE_MODEL_CHAINS:
        chain = get_chain_for_role(role)
        assert len(chain) == 1
        assert chain[0].model_id == "deepseek-ai/DeepSeek-V4.1-Flash-1017"
        assert chain[0].provider == "modal"


def test_extract_thinking_and_answer_uses_explicit_answer_boundary():
    raw = """Answer
For the <answer section: complete polished response with citations and markdown.

Let me craft the response first:
This is internal drafting text.

//the model answer should began from this line the above section should be included in reasoning

### Key Findings
The study reports a grounded result [Source 1].
"""

    thinking, answer = llm_service.extract_thinking_and_answer(raw)

    assert "internal drafting text" in thinking
    assert answer.startswith("### Key Findings")
    assert "internal drafting text" not in answer


def test_extract_thinking_and_answer_hides_untagged_reasoning_without_final_answer():
    _thinking, answer = llm_service.extract_thinking_and_answer(
        "Let me think through the question carefully. The user asks for authors."
    )
    assert answer == ""


def test_strip_think_blocks_handles_unclosed_and_variants():
    assert llm_service._strip_think_blocks(
        "Hello <think>internal monologue</think> world"
    ) == "Hello  world"
    assert llm_service._strip_think_blocks(
        "Answer: 42 <think>unclosed trailing thought"
    ) == "Answer: 42"
    assert llm_service._strip_think_blocks(
        "Final: <thought>reasoning</thought> done"
    ) == "Final:  done"
    assert llm_service._strip_think_blocks(
        "Intro <thinking>deep thinking</thinking> outro"
    ) == "Intro  outro"
