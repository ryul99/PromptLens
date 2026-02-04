from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from promptlens.config import AppConfig
from promptlens.logging_jsonl import JsonlLogger, safe_json_loads, truncate_bytes


def _get_content_type(request_path: str, request_json: Any | None) -> str:
    if not isinstance(request_json, dict):
        return "unknown"

    lowered = request_path.lower()
    if "/chat/completions" in lowered:
        return "chat"
    if "/completions" in lowered:
        return "completion"
    if "/embeddings" in lowered:
        return "embedding"
    if "/images/generations" in lowered or "/images" in lowered:
        return "image"
    if "/responses" in lowered:
        return "response"

    return "unknown"


HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}


def _filter_headers(headers: httpx.Headers) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for k, v in headers.multi_items():
        lk = k.lower()
        if lk in HOP_BY_HOP_HEADERS or lk == "content-length":
            continue
        out.append((k, v))
    return out


def _apply_upstream_headers(
    response: Response, upstream_headers: httpx.Headers
) -> None:
    for k, v in _filter_headers(upstream_headers):
        if k.lower() == "set-cookie":
            response.headers.append(k, v)
        elif k.lower() == "content-type":
            continue
        else:
            response.headers[k] = v


def _filter_request_headers(request: Request) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    # Use raw headers to preserve duplicates (dict(request.headers) drops them).
    # Decode with latin-1 for a lossless, non-throwing byte-to-str mapping required by httpx.
    for k_raw, v_raw in request.headers.raw:
        k = k_raw.decode("latin-1")
        v = v_raw.decode("latin-1")
        lk = k.lower()
        if lk in {"host", "content-length"}:
            continue
        out.append((k, v))
    return out


def _should_stream(request_json: Any | None) -> bool:
    if not isinstance(request_json, dict):
        return False
    return bool(request_json.get("stream") is True)


def _extract_user_input(
    request_path: str, request_json: Any | None
) -> dict[str, Any] | None:
    if not isinstance(request_json, dict):
        return None

    content_type = _get_content_type(request_path, request_json)
    content = _extract_prompt(request_path, request_json)

    return {
        "role": "user",
        "type": content_type,
        "content": content,
    }


def _extract_model_response(
    response_json: Any | None, request_path: str
) -> dict[str, Any] | None:
    if not isinstance(response_json, dict):
        return None

    content_type = _get_content_type(request_path, response_json)
    content = None
    tool_calls = None
    refusal = None

    lowered = request_path.lower()

    if "/chat/completions" in lowered:
        choices = response_json.get("choices", [])
        if choices and isinstance(choices, list):
            first_choice = choices[0]
            if isinstance(first_choice, dict):
                message = first_choice.get("message")
                if isinstance(message, dict):
                    content = message.get("content")
                    tool_calls = message.get("tool_calls")
                    refusal = message.get("refusal")
    elif "/completions" in lowered:
        choices = response_json.get("choices", [])
        if choices and isinstance(choices, list):
            first_choice = choices[0]
            if isinstance(first_choice, dict):
                content = first_choice.get("text")
    elif "/embeddings" in lowered:
        data = response_json.get("data", [])
        if data and isinstance(data, list):
            content = f"embedding with {len(data[0].get('embedding', []))} dimensions"
    elif "/images" in lowered:
        data = response_json.get("data", [])
        if data and isinstance(data, list):
            first_item = data[0]
            if isinstance(first_item, dict):
                content = {
                    "url": first_item.get("url"),
                    "revised_prompt": first_item.get("revised_prompt"),
                }
    else:
        for key in ("content", "text", "output", "result"):
            if key in response_json:
                content = response_json.get(key)
                break

    result: dict[str, Any] = {
        "role": "assistant",
        "type": content_type,
        "content": content,
    }

    if tool_calls:
        result["tool_calls"] = tool_calls
    if refusal:
        result["refusal"] = refusal

    return result


def _parse_streaming_chat_completion(data: bytes) -> dict[str, Any] | None:
    content_parts: list[str] = []
    tool_calls_parts: dict[str, dict[str, Any]] = {}

    try:
        text = data.decode("utf-8", errors="ignore")
    except Exception:
        return None

    for line in text.split("\n"):
        line = line.strip()
        if not line or not line.startswith("data: "):
            continue

        data_str = line[6:].strip()
        if data_str == "[DONE]":
            continue

        try:
            chunk = json.loads(data_str)
        except Exception:
            continue

        if not isinstance(chunk, dict):
            continue

        choices = chunk.get("choices", [])
        if choices and isinstance(choices, list):
            first_choice = choices[0]
            if isinstance(first_choice, dict):
                delta = first_choice.get("delta")
                if isinstance(delta, dict):
                    if "content" in delta and delta["content"]:
                        content_parts.append(delta["content"])

                    if "tool_calls" in delta:
                        tool_calls_delta = delta["tool_calls"]
                        if isinstance(tool_calls_delta, list):
                            for tc in tool_calls_delta:
                                if isinstance(tc, dict):
                                    index = tc.get("index")
                                    if index is not None:
                                        idx_str = str(index)
                                        if idx_str not in tool_calls_parts:
                                            tool_calls_parts[idx_str] = {
                                                "index": index,
                                                "function": {},
                                            }
                                        tc_part = tool_calls_parts[idx_str]

                                        if "id" in tc and tc["id"]:
                                            tc_part["id"] = tc["id"]
                                        if "type" in tc:
                                            tc_part["type"] = tc["type"]

                                        function = tc.get("function")
                                        if isinstance(function, dict):
                                            if "name" in function:
                                                tc_part["function"]["name"] = (
                                                    function.get("name")
                                                )
                                            if "arguments" in function:
                                                func_args = tc_part["function"].get(
                                                    "arguments", ""
                                                )
                                                tc_part["function"]["arguments"] = (
                                                    func_args + function["arguments"]
                                                )

    result: dict[str, Any] = {}
    if content_parts:
        result["content"] = "".join(content_parts)
    if tool_calls_parts:
        result["tool_calls"] = list(tool_calls_parts.values())

    return result if result else None


def _extract_prompt(request_path: str, request_json: Any | None) -> Any | None:
    if not isinstance(request_json, dict):
        return None

    lowered = request_path.lower()
    if "/chat/completions" in lowered:
        return request_json.get("messages")
    if "/responses" in lowered:
        if "input" in request_json:
            return request_json.get("input")
        if "messages" in request_json:
            return request_json.get("messages")
        return None
    if "/completions" in lowered:
        return request_json.get("prompt")
    if "/embeddings" in lowered:
        return request_json.get("input")
    if "/images" in lowered:
        return request_json.get("prompt")

    for key in ("messages", "input", "prompt"):
        if key in request_json:
            return request_json.get(key)
    return None


def _prompt_for_log(
    *,
    prompt: Any | None,
    max_bytes: int,
) -> tuple[Any | None, bool]:
    if prompt is None:
        return None, False

    try:
        encoded = json.dumps(prompt, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    except Exception:
        encoded = str(prompt).encode("utf-8", "replace")

    if len(encoded) <= max_bytes:
        return prompt, False
    truncated, _ = truncate_bytes(encoded, max_bytes)
    return truncated.decode("utf-8", "replace"), True


def create_app(cfg: AppConfig, logger: JsonlLogger) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        existing = getattr(app.state, "http", None)
        if existing is not None:
            yield
            return

        timeout = httpx.Timeout(cfg.upstream.timeout_s)
        app.state.http = httpx.AsyncClient(
            timeout=timeout,
            verify=cfg.upstream.verify_ssl,
            headers=cfg.upstream.headers,
        )
        try:
            yield
        finally:
            await app.state.http.aclose()

    app = FastAPI(title="PromptLens Proxy", version="0.1.0", lifespan=lifespan)

    @app.api_route(
        "/{full_path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
    )
    async def proxy(full_path: str, request: Request) -> Response:
        upstream_url = (
            f"{cfg.upstream.base_url}/{full_path}"
            if full_path
            else cfg.upstream.base_url
        )
        method = request.method.upper()
        request_path = "/" + full_path if full_path else "/"

        body = await request.body()
        req_json = safe_json_loads(body)
        streaming = _should_stream(req_json)

        user_input = _extract_user_input(request_path, req_json)
        if user_input:
            input_for_log, input_truncated = _prompt_for_log(
                prompt=user_input, max_bytes=cfg.logging.max_prompt_bytes
            )
            if input_for_log is not None:
                await logger.write_event(
                    {"input": input_for_log, "truncated": input_truncated}
                )

        req_headers = _filter_request_headers(request)
        query_params_for_upstream = list(request.query_params.multi_items())

        try:
            if streaming:
                return await _proxy_streaming(
                    app=app,
                    cfg=cfg,
                    logger=logger,
                    request_path=request_path,
                    upstream_url=upstream_url,
                    method=method,
                    req_headers=req_headers,
                    query_params_for_upstream=query_params_for_upstream,
                    body=body,
                )

            upstream_resp = await app.state.http.request(
                method,
                upstream_url,
                params=query_params_for_upstream,
                content=body if body else None,
                headers=req_headers,
            )

            resp_json = safe_json_loads(upstream_resp.content)
            model_response = _extract_model_response(resp_json, request_path)
            if model_response:
                response_for_log, response_truncated = _prompt_for_log(
                    prompt=model_response, max_bytes=cfg.logging.max_prompt_bytes
                )
                if response_for_log is not None:
                    await logger.write_event(
                        {"output": response_for_log, "truncated": response_truncated}
                    )

            resp = Response(
                content=upstream_resp.content,
                status_code=upstream_resp.status_code,
                media_type=upstream_resp.headers.get("content-type"),
            )
            _apply_upstream_headers(resp, upstream_resp.headers)
            return resp
        except httpx.RequestError as exc:
            return JSONResponse(
                status_code=502,
                content={
                    "error": {
                        "message": "Upstream request failed",
                        "type": type(exc).__name__,
                    }
                },
            )

    return app


async def _proxy_streaming(
    *,
    app: FastAPI,
    cfg: AppConfig,
    logger: JsonlLogger,
    request_path: str,
    upstream_url: str,
    method: str,
    req_headers: list[tuple[str, str]],
    query_params_for_upstream: list[tuple[str, str]],
    body: bytes,
) -> Response:
    try:
        upstream_cm = app.state.http.stream(
            method,
            upstream_url,
            params=query_params_for_upstream,
            content=body if body else None,
            headers=req_headers,
        )
        upstream_resp = await upstream_cm.__aenter__()
    except httpx.RequestError as exc:
        return JSONResponse(
            status_code=502,
            content={
                "error": {
                    "message": "Upstream request failed",
                    "type": type(exc).__name__,
                }
            },
        )

    status_code = upstream_resp.status_code
    accumulated_chunks: list[bytes] = []

    async def iterator() -> AsyncIterator[bytes]:
        try:
            async for chunk in upstream_resp.aiter_raw():
                yield chunk
                accumulated_chunks.append(chunk)
        finally:
            await upstream_cm.__aexit__(None, None, None)

    async def logging_iterator() -> AsyncIterator[bytes]:
        try:
            async for chunk in iterator():
                yield chunk

            full_bytes = b"".join(accumulated_chunks)
            content_type = _get_content_type(request_path, None)

            model_response = {
                "role": "assistant",
                "type": content_type,
                "content": full_bytes.decode("utf-8", errors="ignore"),
            }

            if "/chat/completions" in request_path.lower():
                parsed = _parse_streaming_chat_completion(full_bytes)
                if parsed:
                    if "content" in parsed:
                        model_response["content"] = parsed["content"]
                    if "tool_calls" in parsed:
                        model_response["tool_calls"] = parsed["tool_calls"]

            response_for_log, response_truncated = _prompt_for_log(
                prompt=model_response, max_bytes=cfg.logging.max_prompt_bytes
            )
            if response_for_log is not None:
                await logger.write_event(
                    {"output": response_for_log, "truncated": response_truncated}
                )
        except Exception:
            pass

    resp = StreamingResponse(
        logging_iterator(),
        status_code=status_code,
        media_type=upstream_resp.headers.get("content-type"),
    )
    _apply_upstream_headers(resp, upstream_resp.headers)
    return resp
