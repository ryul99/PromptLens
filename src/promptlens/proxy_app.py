from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from promptlens.config import AppConfig
from promptlens.logging_jsonl import JsonlLogger, safe_json_loads, truncate_bytes

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

        prompt = _extract_prompt(request_path, req_json)
        prompt_for_log, prompt_truncated = _prompt_for_log(
            prompt=prompt, max_bytes=cfg.logging.max_prompt_bytes
        )
        if prompt_for_log is not None:
            await logger.write_event(
                {"prompt": prompt_for_log, "truncated": prompt_truncated}
            )

        req_headers = _filter_request_headers(request)
        query_params_for_upstream = list(request.query_params.multi_items())

        try:
            if streaming:
                return await _proxy_streaming(
                    app=app,
                    cfg=cfg,
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

    async def iterator() -> AsyncIterator[bytes]:
        try:
            async for chunk in upstream_resp.aiter_raw():
                yield chunk
        finally:
            await upstream_cm.__aexit__(None, None, None)

    resp = StreamingResponse(
        iterator(),
        status_code=status_code,
        media_type=upstream_resp.headers.get("content-type"),
    )
    _apply_upstream_headers(resp, upstream_resp.headers)
    return resp
