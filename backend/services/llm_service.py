from __future__ import annotations

import asyncio
import json
import logging
import hashlib
import re
import time
from collections.abc import Awaitable, Callable
from typing import AsyncIterator
from weakref import WeakKeyDictionary

import httpx

from config import settings
from services.llm_models import ModelSpec
from services import request_context, tracking

logger = logging.getLogger(__name__)
ProgressCallback = Callable[[str], Awaitable[None]]
_MAX_TRANSIENT_RETRIES = 1
_TRANSIENT_RETRY_DELAY_SECONDS = 0.35
_OUTPUT_LIMIT_MARKER = "[The response reached the model output limit and may be incomplete.]"


class NonRetryableLLMError(Exception):
    """Raised when retrying the configured Modal model will not help."""


_semaphores: dict[str, asyncio.Semaphore] = {}
_clients_by_loop: WeakKeyDictionary[
    asyncio.AbstractEventLoop, dict[str, httpx.AsyncClient]
] = WeakKeyDictionary()
_CHAT_RESPONSE_SYSTEM = (
    "Return only the final user-facing answer. Do not include private reasoning, "
    "chain-of-thought, analysis, deliberation, drafting notes, or a reasoning summary. "
    "Answer directly using the supplied evidence, and preserve useful citations and markdown."
)
_AGENT_RESPONSE_SYSTEM = (
    "Return only the final output in the exact format requested. Never include private "
    "reasoning, chain-of-thought, analysis, deliberation, or drafting notes. Ground factual "
    "claims only in the supplied evidence; do not guess or add unsupported details. If the "
    "evidence does not establish a detail, state that it is not specified."
)


def _model_semaphore(model_id: str, max_concurrent: int) -> asyncio.Semaphore:
    if model_id not in _semaphores:
        _semaphores[model_id] = asyncio.Semaphore(max(1, max_concurrent))
    return _semaphores[model_id]


def _get_client(kind: str) -> httpx.AsyncClient:
    """Reuse Modal API connections per event loop."""
    loop = asyncio.get_running_loop()
    clients = _clients_by_loop.setdefault(loop, {})
    client = clients.get(kind)
    if client is None or getattr(client, "is_closed", False):
        timeout = httpx.Timeout(
            connect=settings.MODAL_CONNECT_TIMEOUT,
            read=settings.MODAL_READ_TIMEOUT,
            write=30.0,
            pool=30.0,
        )
        client = httpx.AsyncClient(timeout=timeout)
        clients[kind] = client
    return client


async def close_llm_client() -> None:
    """Close pooled Modal API connections during application shutdown."""
    loop = asyncio.get_running_loop()
    for client in _clients_by_loop.pop(loop, {}).values():
        if not getattr(client, "is_closed", False):
            await client.aclose()


async def check_provider_readiness() -> dict:
    """Check Modal generation access and the Ollama embedding model independently."""
    timeout = httpx.Timeout(connect=2.0, read=3.0, write=3.0, pool=2.0)
    available: list[str] = []
    missing: list[str] = []
    errors: list[str] = []

    if not settings.MODAL_PROXY_TOKEN_ID or not settings.MODAL_PROXY_TOKEN_SECRET:
        missing.append(f"modal:{settings.MODAL_MODEL}")
        errors.append(
            "MODAL_PROXY_TOKEN_ID and MODAL_PROXY_TOKEN_SECRET must be configured."
        )
    else:
        try:
            response = await _get_client("modal").get(
                f"{settings.MODAL_BASE_URL}/models",
                headers=_modal_headers(),
                timeout=timeout,
            )
            response.raise_for_status()
            payload = response.json()
            models = payload.get("data") if isinstance(payload, dict) else None
            names = {
                model.get("id")
                for model in (models if isinstance(models, list) else [])
                if isinstance(model, dict) and isinstance(model.get("id"), str)
            }
            if settings.MODAL_MODEL in names:
                available.append(f"modal:{settings.MODAL_MODEL}")
            else:
                missing.append(f"modal:{settings.MODAL_MODEL}")
        except httpx.HTTPStatusError as error:
            missing.append(f"modal:{settings.MODAL_MODEL}")
            errors.append(f"Modal API returned HTTP {error.response.status_code}.")
        except httpx.RequestError as error:
            missing.append(f"modal:{settings.MODAL_MODEL}")
            errors.append(f"Could not reach Modal API ({type(error).__name__}).")
        except ValueError:
            missing.append(f"modal:{settings.MODAL_MODEL}")
            errors.append("Modal API returned an invalid model inventory.")

    try:
        async with httpx.AsyncClient(timeout=timeout) as ollama_client:
            response = await ollama_client.get(
                f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/tags",
            )
            response.raise_for_status()
            payload = response.json()
    except httpx.HTTPStatusError as error:
        missing.append(f"ollama:{settings.OLLAMA_EMBED_MODEL}")
        errors.append(f"Ollama returned HTTP {error.response.status_code}.")
    except httpx.RequestError as error:
        missing.append(f"ollama:{settings.OLLAMA_EMBED_MODEL}")
        errors.append(f"Could not reach Ollama embeddings ({type(error).__name__}).")
    except ValueError:
        missing.append(f"ollama:{settings.OLLAMA_EMBED_MODEL}")
        errors.append("Ollama returned an invalid model inventory.")
    else:
        models = payload.get("models") if isinstance(payload, dict) else None
        names = {
            model.get("name")
            for model in (models if isinstance(models, list) else [])
            if isinstance(model, dict) and isinstance(model.get("name"), str)
        }
        if (
            settings.OLLAMA_EMBED_MODEL in names
            or f"{settings.OLLAMA_EMBED_MODEL}:latest" in names
        ):
            available.append(f"ollama:{settings.OLLAMA_EMBED_MODEL}")
        else:
            missing.append(f"ollama:{settings.OLLAMA_EMBED_MODEL}")

    return {
        "status": "ok" if not missing else "not_ready",
        "available_models": available,
        "missing_models": missing,
        "errors": errors,
        "inference_tested": False,
    }


def _should_retry_http_status(status_code: int) -> bool:
    return status_code == 429 or status_code >= 500


async def _post_with_retry(
    client: httpx.AsyncClient,
    url: str,
    payload: dict,
    *,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    for attempt in range(_MAX_TRANSIENT_RETRIES + 1):
        try:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as error:
            if (
                not _should_retry_http_status(error.response.status_code)
                or attempt >= _MAX_TRANSIENT_RETRIES
            ):
                raise
        except httpx.ReadTimeout:
            raise
        except httpx.RequestError:
            if attempt >= _MAX_TRANSIENT_RETRIES:
                raise
        await asyncio.sleep(_TRANSIENT_RETRY_DELAY_SECONDS * (attempt + 1))
    raise RuntimeError("Modal request retry loop exited unexpectedly.")


_UNTAGGED_REASONING_PREFIXES = (
    "answer\nfor the <answer section",
    "alright, let me",
    "alright, let me tackle this user query",
    "alright let me tackle this user query",
    "we are given a user question",
    "let me think",
    "let me reason",
    "let me analyze",
    "let me break this down",
    "let's think",
    "let's reason",
    "let's analyze",
    "reasoning:",
    "analysis:",
    "i need to answer",
    "i should answer",
    "we need to answer",
    "we need to analyze",
    "we need to reason",
    "i will analyze",
    "i'll analyze",
    "first, i need to",
    "first i need to",
    "looking at the document excerpts",
    "let me check the document",
    "let's analyze the question",
    "the user is asking",
    "the user wants to know",
    "important: the user is asking",
    "the user's question is",
    "given the strict rules",
    "let me draft",
)
_UNTAGGED_ANSWER_BOUNDARY = re.compile(
    r"(?im)^\s*(?://\s*)?the model answer should beg(?:in|an) from this line"
    r"[^\r\n]*(?:\r?\n+|$)"
)
_UNTAGGED_ANSWER_MARKER = re.compile(
    r"(?:^|\n+)\s*(?:(?:Concise|Descriptive)\s+Grounded\s+Answer|"
    r"Final(?:\s+(?:draft|version|answer))?|Answer)\s*:\s*",
    re.IGNORECASE,
)


def _find_untagged_answer_start(text: str) -> re.Match | None:
    boundary = _UNTAGGED_ANSWER_BOUNDARY.search(text)
    if boundary:
        return boundary
    marker = _UNTAGGED_ANSWER_MARKER.search(text)
    if marker:
        return marker
    return re.search(
        r"\n{2,}(?=(?:#{1,3}\s|\*\*|Based on\b|In summary\b|According to\b|"
        r"The answer is\b|The main (?:conclusions|findings)\b))",
        text,
        flags=re.IGNORECASE,
    )


def extract_thinking_and_answer(text: str) -> tuple[str, str]:
    """Separate a user-facing rationale summary from the final answer.

    Handles both explicit <think>...</think> tags and untagged reasoning preambles
    (e.g., 'We are given a user question: ...' followed by 'Grounded Answer:').
    Also handles explicit <summary> and <answer> sections. Returns (summary, answer).
    """
    if not text:
        return "", ""
    raw = text.strip()
    thinking_parts: list[str] = []

    summary_matches = list(re.finditer(
        r"<summary>([\s\S]*?)(?:</summary>|(?=<answer>)|$)",
        raw,
        flags=re.IGNORECASE,
    ))
    answer_matches = list(re.finditer(
        r"<answer>([\s\S]*?)(?:</answer>|$)",
        raw,
        flags=re.IGNORECASE,
    ))
    if summary_matches or answer_matches:
        summary = "\n\n".join(
            match.group(1).strip() for match in summary_matches if match.group(1).strip()
        )
        answer = "\n\n".join(
            match.group(1).strip() for match in answer_matches if match.group(1).strip()
        )
        if not answer and not summary:
            answer = re.sub(r"</?(?:summary|answer)>", "", raw, flags=re.IGNORECASE).strip()
        answer = re.sub(
            r"<(?:think|thought|thinking)>[\s\S]*?(?:</(?:think|thought|thinking)>|$)",
            "",
            answer,
            flags=re.IGNORECASE,
        ).strip()
        return summary, answer

    boundary_match = _UNTAGGED_ANSWER_BOUNDARY.search(raw)
    if boundary_match:
        return raw[:boundary_match.start()].strip(), raw[boundary_match.end():].strip()

    # 1. Extract tagged <think> blocks
    for m in re.finditer(
        r"<(?:think|thought|thinking)>([\s\S]*?)(?:</(?:think|thought|thinking)>|$)",
        raw,
        re.IGNORECASE,
    ):
        if m.group(1).strip():
            thinking_parts.append(m.group(1).strip())

    cleaned = re.sub(
        r"<(?:think|thought|thinking)>.*?(?:</(?:think|thought|thinking)>|$)",
        "",
        raw,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()
    cleaned = re.sub(r"</(?:think|thought|thinking)>", "", cleaned, flags=re.IGNORECASE).strip()

    # 2. Extract untagged reasoning preamble if an answer marker is present
    has_reasoning_preamble = cleaned.lstrip().lower().startswith(_UNTAGGED_REASONING_PREFIXES)
    marker_match = _find_untagged_answer_start(cleaned) if has_reasoning_preamble else None
    if marker_match and marker_match.start() > 0 and has_reasoning_preamble:
        preamble = cleaned[:marker_match.start()].strip()
        if preamble:
            thinking_parts.append(preamble)
        cleaned = cleaned[marker_match.end():].strip()
    elif has_reasoning_preamble:
        answer_start = _find_untagged_answer_start(cleaned)
        if answer_start:
            thinking_parts.append(cleaned[:answer_start.start()].strip())
            cleaned = cleaned[answer_start.end():].strip()
        else:
            thinking_parts.append(cleaned)
            cleaned = ""

    thinking = "\n\n".join(thinking_parts).strip()
    return thinking, cleaned


def _modal_payload(
    messages: list[dict],
    *,
    temperature: float,
    max_tokens: int,
    reasoning_effort: str | None = None,
    json_mode: bool = False,
    stream: bool = False,
    chat_mode: bool = False,
) -> dict:
    request_messages = [dict(message) for message in messages]
    if chat_mode:
        system_message = next(
            (message for message in request_messages if message.get("role") == "system"),
            None,
        )
        if system_message is None:
            request_messages.insert(0, {"role": "system", "content": _CHAT_RESPONSE_SYSTEM})
        else:
            system_message["content"] = (
                f"{system_message['content']}\n\n{_CHAT_RESPONSE_SYSTEM}"
            )
    payload = {
        "model": settings.MODAL_MODEL,
        "messages": request_messages,
        "temperature": temperature,
        "max_tokens": max(1, min(max_tokens, settings.MODAL_MAX_TOKENS)),
        "top_p": settings.MODAL_TOP_P,
        "reasoning_effort": reasoning_effort or settings.MODAL_REASONING_EFFORT,
        "stream": stream,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    return payload


def _modal_model_spec() -> ModelSpec:
    return ModelSpec(
        "modal",
        settings.MODAL_MODEL,
        max_concurrent=settings.MODAL_MAX_CONCURRENT,
        num_ctx=settings.MODAL_CONTEXT,
        provider="modal",
    )


def _modal_headers() -> dict[str, str]:
    if not settings.MODAL_PROXY_TOKEN_ID or not settings.MODAL_PROXY_TOKEN_SECRET:
        raise NonRetryableLLMError(
            "MODAL_PROXY_TOKEN_ID and MODAL_PROXY_TOKEN_SECRET must be configured."
        )
    return {
        "Authorization": (
            f"Bearer {settings.MODAL_PROXY_TOKEN_ID}."
            f"{settings.MODAL_PROXY_TOKEN_SECRET}"
        ),
        "Accept": "application/json",
    }


def _modal_response(data: dict) -> tuple[str, dict, str | None]:
    choices = data.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices else None
    message = choice.get("message") if isinstance(choice, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    finish_reason = choice.get("finish_reason") if isinstance(choice, dict) else None
    usage_data = data.get("usage")
    usage_data = usage_data if isinstance(usage_data, dict) else {}
    if not isinstance(content, str) or not content.strip():
        if finish_reason == "length":
            completion_tokens = usage_data.get("completion_tokens")
            token_detail = (
                f" after using {completion_tokens} completion tokens"
                if isinstance(completion_tokens, int)
                else ""
            )
            raise NonRetryableLLMError(
                "Modal exhausted the output token limit before returning answer text"
                f"{token_detail}. Increase the request max_tokens or lower "
                "MODAL_REASONING_EFFORT."
            )
        raise NonRetryableLLMError("Modal API returned an empty response.")
    usage = {
        "prompt_eval_count": usage_data.get("prompt_tokens", 0),
        "eval_count": usage_data.get("completion_tokens", 0),
    }
    return content, usage, finish_reason


async def _call_modal_once(
    messages: list[dict],
    *,
    temperature: float,
    max_tokens: int,
    json_mode: bool,
    reasoning_effort: str | None = None,
    chat_mode: bool = False,
) -> tuple[str, dict]:
    payload = _modal_payload(
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        json_mode=json_mode,
        chat_mode=chat_mode,
    )
    url = f"{settings.MODAL_BASE_URL}/chat/completions"
    try:
        async with _semaphore(_modal_model_spec()):
            response = await _post_with_retry(
                _get_client("modal"),
                url,
                payload,
                headers=_modal_headers(),
            )
            data = response.json()
    except httpx.ReadTimeout as error:
        raise NonRetryableLLMError(
            "The Modal request timed out while waiting for model output. "
            "It was not retried to avoid duplicate generation."
        ) from error

    content, usage, finish_reason = _modal_response(data)
    content = _strip_think_blocks(content)
    if not content:
        raise NonRetryableLLMError("Modal API returned no user-facing response.")
    if finish_reason == "length":
        content = f"{content.rstrip()}\n\n[The response reached the model output limit and may be incomplete.]"
    return content, usage


async def _call_modal_chat_once(
    messages: list[dict],
    *,
    temperature: float,
    max_tokens: int,
) -> tuple[str, dict]:
    return await _call_modal_once(
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
        json_mode=False,
        chat_mode=True,
    )


def _partial_tag_suffix_length_any(text: str, tags: tuple[str, ...]) -> int:
    lowered = text.lower()
    for length in range(min(len(text), max(map(len, tags)) - 1), 0, -1):
        if any(lowered.endswith(tag[:length]) for tag in tags):
            return length
    return 0


def _filter_chat_events(
    content: str,
    state: dict,
    *,
    final: bool = False,
) -> list[tuple[str, str]]:
    """Emit only answer text; discard summary and private thought channels."""
    open_tags = {
        "<summary>": "summary",
        "<answer>": "answer",
        "<think>": "private",
        "<thought>": "private",
        "<thinking>": "private",
    }
    close_tags = {
        "</summary>": "summary",
        "</answer>": "answer",
        "</think>": "private",
        "</thought>": "private",
        "</thinking>": "private",
    }
    all_tags = tuple(open_tags) + tuple(close_tags)
    data = state.get("pending", "") + content
    events: list[tuple[str, str]] = []
    has_section_tags = bool(
        re.search(r"<(?:summary|answer|think|thought|thinking)>", data, flags=re.IGNORECASE)
    )
    if "mode" not in state and not has_section_tags:
        stripped = data.lstrip().lower()
        if stripped.startswith(_UNTAGGED_REASONING_PREFIXES):
            state["untagged_reasoning"] = True
        elif any(prefix.startswith(stripped) for prefix in _UNTAGGED_REASONING_PREFIXES):
            if not final:
                state["pending"] = data
                return events
            state["untagged_reasoning"] = True
        else:
            state["mode"] = "answer"
            state["answer_seen"] = True

    if state.get("untagged_reasoning"):
        answer_marker = _find_untagged_answer_start(data)
        answer_tag = re.search(r"<answer>", data, flags=re.IGNORECASE)
        if answer_tag and (answer_marker is None or answer_tag.start() < answer_marker.start()):
            answer_marker = answer_tag
        if answer_marker is None:
            state["pending"] = "" if final else data
            return events
        data = data[answer_marker.end():]
        state["untagged_reasoning"] = False
        state["mode"] = "answer"
        state["answer_seen"] = True
        state["pending"] = ""

    def emit(text: str) -> None:
        mode = state.get("mode", "discard")
        if text and mode == "answer":
            events.append(("token", text))
            if mode == "answer":
                state["answer_emitted"] = True

    while data:
        lowered = data.lower()
        matches = [
            (lowered.find(tag), tag)
            for tag in all_tags
            if lowered.find(tag) >= 0
        ]
        if not matches:
            if final:
                emit(data)
                state["pending"] = ""
                return events
            suffix_length = _partial_tag_suffix_length_any(data, all_tags)
            emit(data[:-suffix_length] if suffix_length else data)
            state["pending"] = data[-suffix_length:] if suffix_length else ""
            return events

        position, tag = min(matches, key=lambda match: match[0])
        emit(data[:position])
        data = data[position + len(tag):]
        if tag in open_tags:
            state["mode"] = open_tags[tag]
            if state["mode"] == "answer":
                state["answer_seen"] = True
        elif state.get("mode") == close_tags[tag]:
            state["mode"] = (
                "answer"
                if tag == "</summary>" or close_tags[tag] == "private"
                else "discard"
            )
            if state["mode"] == "answer":
                state["answer_seen"] = True
    return events


async def _call_modal_chat_stream(
    messages: list[dict],
    *,
    temperature: float,
    max_tokens: int,
) -> AsyncIterator[tuple[str, str]]:
    payload = _modal_payload(
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
        chat_mode=True,
    )
    url = f"{settings.MODAL_BASE_URL}/chat/completions"

    async with _semaphore(_modal_model_spec()):
        for attempt in range(_MAX_TRANSIENT_RETRIES + 1):
            state: dict = {}
            emitted = False
            response_limit_reached = False
            try:
                async with _get_client("modal").stream(
                    "POST",
                    url,
                    json=payload,
                    headers=_modal_headers(),
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line or line.startswith(":") or not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError as error:
                            raise NonRetryableLLMError(
                                "Modal API returned an invalid streaming response."
                            ) from error
                        emitted = True
                        choices = chunk.get("choices")
                        choice = choices[0] if isinstance(choices, list) and choices else {}
                        delta = choice.get("delta") if isinstance(choice, dict) else None
                        content = delta.get("content") if isinstance(delta, dict) else None
                        if isinstance(content, str) and content:
                            for event in _filter_chat_events(content, state):
                                emitted = True
                                yield event
                        if isinstance(choice, dict) and choice.get("finish_reason") == "length":
                            response_limit_reached = True
                    for event in _filter_chat_events("", state, final=True):
                        emitted = True
                        yield event
                if not emitted or not state.get("answer_seen"):
                    raise NonRetryableLLMError("Modal API returned no user-facing chat response.")
                if response_limit_reached:
                    yield (
                        "warning",
                        "The model reached its output limit. This answer may be incomplete.",
                    )
                return
            except httpx.HTTPStatusError as error:
                if (
                    not emitted
                    and _should_retry_http_status(error.response.status_code)
                    and attempt < _MAX_TRANSIENT_RETRIES
                ):
                    logger.warning(
                        "modal_chat_stream_retry",
                        extra={"status_code": error.response.status_code, "attempt": attempt + 1},
                    )
                    await asyncio.sleep(_TRANSIENT_RETRY_DELAY_SECONDS * (attempt + 1))
                    continue
                raise NonRetryableLLMError(
                    f"Modal chat request failed with HTTP {error.response.status_code}."
                ) from error
            except httpx.ReadTimeout as error:
                raise NonRetryableLLMError(
                    "The Modal chat request timed out while waiting for model output. "
                    "The request was not retried to avoid duplicate generation."
                ) from error
            except httpx.RequestError as error:
                if not emitted and attempt < _MAX_TRANSIENT_RETRIES:
                    logger.warning(
                        "modal_chat_stream_retry",
                        extra={"error_type": type(error).__name__, "attempt": attempt + 1},
                    )
                    await asyncio.sleep(_TRANSIENT_RETRY_DELAY_SECONDS * (attempt + 1))
                    continue
                message = (
                    "Cannot reach the Modal API. Check MODAL_BASE_URL and network access."
                    if isinstance(error, httpx.ConnectError)
                    else "The Modal chat request failed or timed out. Try again or use a shorter question."
                )
                raise NonRetryableLLMError(message) from error


def _strip_think_blocks(text: str) -> str:
    return extract_thinking_and_answer(text)[1]


def _semaphore(spec: ModelSpec) -> asyncio.Semaphore:
    return _model_semaphore(spec.model_id, spec.max_concurrent)


_background_tasks: set[asyncio.Task] = set()


def _write_tracking(agent_role: str, model_id: str, prompt: str, usage: dict, latency_ms: int, user_id) -> None:
    """Synchronous Supabase inserts; always run off the event loop."""
    tokens_in = int(usage.get("prompt_eval_count") or 0)
    tokens_out = int(usage.get("eval_count") or 0)
    tracking.log_api_cost(
        agent=agent_role,
        model=model_id,
        tokens_used=tokens_in + tokens_out,
        user_id=user_id,
    )
    tracking.log_audit_event(
        node_name=agent_role,
        model_used=model_id,
        prompt_hash=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        latency_ms=latency_ms,
    )


def _track_in_background(**kwargs) -> None:
    """Fire-and-forget tracking so remote Supabase writes never delay or block an answer."""
    try:
        task = asyncio.get_running_loop().create_task(asyncio.to_thread(_write_tracking, **kwargs))
    except RuntimeError:
        return
    _background_tasks.add(task)
    task.add_done_callback(_finish_tracking_task)


def _finish_tracking_task(task: asyncio.Task) -> None:
    _background_tasks.discard(task)
    if task.cancelled():
        return
    error = task.exception()
    if error is not None:
        logger.warning(
            "llm_tracking_task_failed",
            extra={"error": f"{type(error).__name__}: {error}"},
        )


# Roles whose output is always JSON.
_JSON_ROLES: frozenset[str] = frozenset({
    "sufficiency_evaluator",
    "gap_report",
    "outline_plan",
    "source_quality_evaluator",
    "user_doc_quality_eval",
    "citation_verification",
    "verification",
})


async def call_llm(
    prompt: str,
    *,
    agent_role: str,
    system: str = "",
    temperature: float = 0.1,
    max_tokens: int = 2048,
    reasoning_effort: str | None = None,
    context_limit: int | None = None,
    json_mode: bool | None = None,
    progress_callback: ProgressCallback | None = None,
) -> str:
    """Call the LLM for a given role. json_mode=None auto-detects from _JSON_ROLES."""
    if json_mode is None:
        json_mode = agent_role in _JSON_ROLES
    system = f"{system}\n\n{_AGENT_RESPONSE_SYSTEM}".strip()
    messages = (
        ([{"role": "system", "content": system}] if system else [])
        + [{"role": "user", "content": prompt}]
    )

    if agent_role == "chat":
        answer, _ = await _call_modal_chat_once(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return answer

    spec = _modal_model_spec()
    started_at = time.perf_counter()
    estimate = (len(prompt) + len(system)) / 3.5
    effective_context = context_limit or spec.num_ctx
    if estimate + max_tokens > effective_context:
        logger.warning(
            "llm_prompt_may_exceed_context",
            extra={
                "agent_role": agent_role,
                "estimated_prompt_tokens": round(estimate),
                "max_tokens": max_tokens,
                "context_limit": effective_context,
            },
        )

    logger.info("llm_request", extra={"agent_role": agent_role, "model": spec.model_id})
    effective_reasoning_effort = reasoning_effort or "none"
    if reasoning_effort is None:
        if agent_role in {"summarization", "summarization_final"}:
            effective_reasoning_effort = settings.MODAL_SUMMARIZATION_REASONING_EFFORT
        elif agent_role == "literature_review":
            effective_reasoning_effort = settings.MODAL_LITERATURE_REVIEW_REASONING_EFFORT

    retry_max_tokens = min(
        settings.MODAL_MAX_TOKENS,
        effective_context - round(estimate),
        max(max_tokens + 1, max_tokens * 2),
    )
    retry_max_tokens = max(max_tokens, retry_max_tokens)
    try:
        try:
            result, usage = await _call_modal_once(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=json_mode,
                reasoning_effort=effective_reasoning_effort,
            )
        except NonRetryableLLMError as error:
            if (
                "exhausted the output token limit before returning answer text" not in str(error)
                or retry_max_tokens <= max_tokens
            ):
                raise
            logger.warning(
                "llm_output_limit_retry",
                extra={
                    "agent_role": agent_role,
                    "from_max_tokens": max_tokens,
                    "to_max_tokens": retry_max_tokens,
                    "reasoning_effort": "none",
                },
            )
            result, usage = await _call_modal_once(
                messages,
                temperature=temperature,
                max_tokens=retry_max_tokens,
                json_mode=json_mode,
                reasoning_effort="none",
            )

        if (
            _OUTPUT_LIMIT_MARKER in result
            and retry_max_tokens > max_tokens
        ):
            logger.warning(
                "llm_truncated_output_retry",
                extra={
                    "agent_role": agent_role,
                    "from_max_tokens": max_tokens,
                    "to_max_tokens": retry_max_tokens,
                },
            )
            result, usage = await _call_modal_once(
                messages,
                temperature=temperature,
                max_tokens=retry_max_tokens,
                json_mode=json_mode,
                reasoning_effort="none",
            )
    except Exception as error:
        logger.exception(
            "llm_modal_request_failed",
            extra={"agent_role": agent_role, "model": spec.model_id},
        )
        if progress_callback:
            await progress_callback(
                f"The Modal request for {spec.model_id} failed; no LLM fallback is configured."
            )
        raise RuntimeError(
            f"Modal request failed for role '{agent_role}': {error}"
        ) from error

    logger.info(
        "llm_response",
        extra={
            "agent_role": agent_role,
            "model": spec.model_id,
            "response_length": len(result),
            "latency_ms": int((time.perf_counter() - started_at) * 1000),
        },
    )
    _track_in_background(
        agent_role=agent_role,
        model_id=spec.model_id,
        prompt=prompt,
        usage=usage,
        latency_ms=int((time.perf_counter() - started_at) * 1000),
        user_id=request_context.get_user_id(),
    )
    return result


async def call_llm_stream(
    prompt: str,
    *,
    agent_role: str,
    system: str = "",
    temperature: float = 0.1,
    max_tokens: int = 2048,
    context_limit: int | None = None,
    emit_events: bool = False,
) -> AsyncIterator[any]:
    if agent_role == "chat":
        system = f"{system}\n\n{_AGENT_RESPONSE_SYSTEM}".strip()
        messages = (
            ([{"role": "system", "content": system}] if system else [])
            + [{"role": "user", "content": prompt}]
        )
        async for event_type, chunk in _call_modal_chat_stream(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            if emit_events:
                yield (event_type, chunk)
            elif event_type == "token":
                yield chunk
        return

    messages = (
        ([{"role": "system", "content": system}] if system else [])
        + [{"role": "user", "content": prompt}]
    )

    started_at = time.perf_counter()
    spec = _modal_model_spec()
    estimate = (len(prompt) + len(system)) / 3.5
    effective_context = context_limit or spec.num_ctx
    if estimate + max_tokens > effective_context:
        logger.warning(
            "llm_prompt_may_exceed_context",
            extra={
                "agent_role": agent_role,
                "estimated_prompt_tokens": round(estimate),
                "max_tokens": max_tokens,
                "context_limit": effective_context,
            },
        )
    try:
        result, usage = await _call_modal_once(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=False,
            reasoning_effort="none",
        )
        _track_in_background(
            agent_role=agent_role,
            model_id=settings.MODAL_MODEL,
            prompt=prompt,
            usage=usage,
            latency_ms=int((time.perf_counter() - started_at) * 1000),
            user_id=request_context.get_user_id(),
        )
        if emit_events:
            yield ("token", result)
        else:
            yield result
    except Exception as error:
        logger.exception(
            "llm_stream_modal_request_failed",
            extra={"agent_role": agent_role, "error": f"{type(error).__name__}: {error}"},
        )
        raise RuntimeError(
            f"Modal stream request failed for role '{agent_role}': {error}"
        ) from error
