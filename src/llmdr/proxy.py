"""HTTP proxy for the Anthropic Messages API with mandatory audit capture.

This module implements the live ingestion mode for llmdr. It exposes a
single endpoint, ``POST /v1/messages``, that transparently forwards
requests to the configured upstream (defaults to api.anthropic.com)
while writing every prompt and response pair to the tamper-evident
audit store.

Design notes (see CLAUDE.md and the proxy.py design proposal for the
full rationale):

* **Split write.** A user audit record is written *before* the upstream
  call so a prompt is preserved even when forwarding fails. The
  assistant audit record is written *after* a successful upstream
  response, linked to the user record by ``conversation_id`` and the
  Anthropic ``request-id`` header.
* **Audit failures fail the client.** If either audit write fails, the
  proxy returns 500 to the client rather than silently delivering an
  unaudited response. The forensic contract is absolute, not best
  effort. A withheld response is recoverable; a quiet forensic gap is
  not.
* **No streaming in v0.1.** Requests with ``stream=true`` are rejected
  with an explicit 400. SSE assembly is a v0.2 target.
* **Single endpoint.** Anything other than ``POST /v1/messages`` returns
  an llmdr-shaped 404 so it is unambiguous that the proxy answered,
  not the upstream.
* **Per-write AuditStore.** Each audit write opens a fresh AuditStore
  on the worker thread that performs the write, then closes it. See
  :func:`_write_audit_record` for the reasoning; the short version is
  that sqlite3 connections are thread-affine by default and asyncio's
  default executor does not pin requests to threads.
* **No request_id on the user record.** Even when the upstream response
  is a 4xx or 5xx and Anthropic returned a ``request-id`` header, that
  id is NOT written to the user record. The user record represents the
  attempt, not the failed response, and its ``request_id`` column
  stays None per its documented default in :mod:`llmdr.audit`.

Logging discipline (non-negotiable, on par with the no-log rule for
the encryption key in :mod:`llmdr.config`):

    No statement in this module passes request body bytes, response
    body bytes, parsed JSON, ``response.text``, ``response.json()``,
    captured ``content``, or any derivative of an LLM payload to a
    logger. Log lines may reference status codes, header names,
    exception types, exception messages from connection-level errors,
    conversation ids, model names, and the upstream base URL. They may
    not reference payload content. Content lives in the encrypted
    audit store and nowhere else. Any future log statement added to
    this module must be auditable against this rule on inspection.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator, Final, Literal, Mapping

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from llmdr.audit import AuditStore
from llmdr.config import Config

logger = logging.getLogger(__name__)


MESSAGES_PATH: Final = "/v1/messages"
CONVERSATION_ID_HEADER: Final = "x-llmdr-conversation-id"
AUDITED_RESPONSE_HEADER: Final = "x-llmdr-audited"
UPSTREAM_REQUEST_ID_HEADER: Final = "request-id"

_UPSTREAM_CONNECT_TIMEOUT_SECONDS: Final = 10.0
_UPSTREAM_READ_TIMEOUT_SECONDS: Final = 300.0

# Hop-by-hop headers per RFC 7230 section 6.1, plus Host (httpx sets
# its own Host based on the upstream base URL) and Content-Length
# (recomputed by both httpx outbound and Starlette inbound from the
# actual body bytes). These are stripped in both directions.
_HOP_BY_HOP_HEADERS: Final = frozenset(
    {
        "connection",
        "keep-alive",
        "transfer-encoding",
        "te",
        "trailer",
        "trailers",
        "upgrade",
        "proxy-authenticate",
        "proxy-authorization",
        "host",
        "content-length",
    }
)

# Headers consumed by the proxy itself and never forwarded upstream.
_PROXY_ONLY_REQUEST_HEADERS: Final = frozenset({CONVERSATION_ID_HEADER})

# Sanity bound for the optional client-supplied conversation id.
_MAX_CONVERSATION_ID_LEN: Final = 256


def create_app(config: Config) -> FastAPI:
    """Build the FastAPI proxy app bound to ``config``.

    The returned app owns a single shared :class:`httpx.AsyncClient`
    via its lifespan, keyed to ``config.upstream_base_url``. The
    config itself is attached to ``app.state.config`` so request
    handlers and tests can reach it without import-time globals.

    Tests can override the upstream client by reassigning
    ``app.state.upstream`` between app construction and the first
    request, or by passing :class:`httpx.MockTransport` through a
    bespoke AsyncClient. Tests can also construct a :class:`Config`
    directly without going through the env loader.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        client = httpx.AsyncClient(
            base_url=config.upstream_base_url,
            timeout=httpx.Timeout(
                _UPSTREAM_READ_TIMEOUT_SECONDS,
                connect=_UPSTREAM_CONNECT_TIMEOUT_SECONDS,
            ),
        )
        app.state.upstream = client
        logger.info(
            "llmdr proxy lifespan started, upstream=%s",
            config.upstream_base_url,
        )
        try:
            yield
        finally:
            await client.aclose()
            logger.info("llmdr proxy lifespan ended, upstream client closed")

    # docs_url, redoc_url, openapi_url are disabled because the proxy
    # is a transparent forwarder, not a documented API surface. The
    # catch-all 404 below would intercept these paths anyway; turning
    # them off at the FastAPI layer keeps the routing table honest.
    app = FastAPI(
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.config = config

    @app.post(MESSAGES_PATH)
    async def messages(request: Request) -> Response:
        return await _handle_messages(request)

    # Catch-all: any other path or method returns an llmdr-shaped 404.
    # The body makes it unambiguous that the proxy answered, not the
    # upstream, so operators and clients can diagnose misrouted calls.
    @app.api_route(
        "/{full_path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
    )
    async def unsupported(full_path: str, request: Request) -> JSONResponse:
        return _error_response(
            status_code=404,
            error_type="llmdr_unsupported",
            message=(
                f"llmdr v0.1 only proxies POST {MESSAGES_PATH}; "
                f"{request.method} /{full_path} is not supported."
            ),
        )

    return app


async def _handle_messages(request: Request) -> Response:
    """Implementation of POST /v1/messages.

    Split into a module-level function so it is testable directly with
    a constructed :class:`fastapi.Request`, and so the route
    registration inside :func:`create_app` stays a one-liner.
    """
    config: Config = request.app.state.config
    upstream: httpx.AsyncClient = request.app.state.upstream

    raw_body = await request.body()

    # Decode once to UTF-8 before parsing, so the audit record stores
    # exactly the bytes the client sent and we have a single point of
    # failure for non-UTF-8 input. json.loads tolerates UTF-16 and
    # UTF-32 input by spec, but the Anthropic API is documented as
    # UTF-8 and we hard-require that here to avoid storing audit
    # content under one encoding that was parsed under another.
    try:
        body_text = raw_body.decode("utf-8")
    except UnicodeDecodeError:
        return _error_response(
            status_code=400,
            error_type="llmdr_bad_request",
            message="Request body is not valid UTF-8.",
        )

    try:
        body_obj = json.loads(body_text) if body_text else None
    except json.JSONDecodeError:
        return _error_response(
            status_code=400,
            error_type="llmdr_bad_request",
            message="Request body is not valid JSON.",
        )
    if not isinstance(body_obj, dict):
        return _error_response(
            status_code=400,
            error_type="llmdr_bad_request",
            message="Request body must be a JSON object.",
        )

    # Streaming rejection. See design proposal section 2.
    if body_obj.get("stream"):
        return _error_response(
            status_code=400,
            error_type="llmdr_unsupported",
            message=(
                "llmdr v0.1 does not proxy streaming requests. "
                "Set stream=false."
            ),
        )

    model = body_obj.get("model")
    if not isinstance(model, str) or not model:
        return _error_response(
            status_code=400,
            error_type="llmdr_bad_request",
            message="Request body must include a non-empty 'model' field.",
        )

    conversation_id = _resolve_conversation_id(request.headers)
    if conversation_id is None:
        return _error_response(
            status_code=400,
            error_type="llmdr_bad_request",
            message=(
                f"{CONVERSATION_ID_HEADER} header, when provided, must be "
                f"1 to {_MAX_CONVERSATION_ID_LEN} characters."
            ),
        )

    # Pre-call audit write. Failure here means the prompt never left
    # the proxy, so there is no upstream cost to apologize for and the
    # chain stays empty for this request.
    try:
        await asyncio.to_thread(
            _write_audit_record,
            config=config,
            conversation_id=conversation_id,
            request_id=None,
            role="user",
            content=body_text,
            model=model,
        )
    except Exception:
        logger.exception(
            "audit write failed for user record, conversation_id=%s",
            conversation_id,
        )
        return _error_response(
            status_code=500,
            error_type="llmdr_audit_failure",
            message=(
                "Audit write failed before upstream call; "
                "request was not forwarded. Check proxy logs."
            ),
        )

    upstream_headers = _filter_request_headers(request.headers)
    try:
        upstream_response = await upstream.request(
            "POST",
            MESSAGES_PATH,
            content=raw_body,
            headers=upstream_headers,
        )
    except httpx.RequestError as exc:
        # Connection-level failure (DNS, TLS, timeout, refused). The
        # exception class name and str(exc) are connection-level
        # diagnostics; httpx.RequestError fires before any response
        # body exists, so neither can carry payload content.
        logger.error(
            "upstream request failed: %s (%s), conversation_id=%s",
            type(exc).__name__,
            exc,
            conversation_id,
        )
        return _error_response(
            status_code=502,
            error_type="llmdr_upstream_unreachable",
            message=(
                f"Upstream at {config.upstream_base_url} unreachable: "
                f"{type(exc).__name__}."
            ),
        )

    upstream_status = upstream_response.status_code
    upstream_body = upstream_response.content
    upstream_request_id = upstream_response.headers.get(
        UPSTREAM_REQUEST_ID_HEADER
    )

    if 200 <= upstream_status < 300:
        try:
            assistant_content = upstream_body.decode("utf-8")
        except UnicodeDecodeError:
            logger.error(
                "upstream returned non-UTF-8 body on status=%d, "
                "conversation_id=%s",
                upstream_status,
                conversation_id,
            )
            return _error_response(
                status_code=500,
                error_type="llmdr_upstream_invalid",
                message=(
                    "Upstream returned a non-UTF-8 response body; "
                    "audit write would be unsafe and response is withheld."
                ),
            )

        try:
            await asyncio.to_thread(
                _write_audit_record,
                config=config,
                conversation_id=conversation_id,
                request_id=upstream_request_id,
                role="assistant",
                content=assistant_content,
                model=model,
            )
        except Exception:
            logger.exception(
                "audit write failed for assistant record after upstream "
                "success, conversation_id=%s, upstream_status=%d, "
                "withholding response from client",
                conversation_id,
                upstream_status,
            )
            return _error_response(
                status_code=500,
                error_type="llmdr_audit_failure",
                message=(
                    "Upstream call succeeded but audit write failed; "
                    "response not delivered. Check proxy logs."
                ),
            )
    else:
        # Upstream-level error (4xx, 5xx). The user record stands
        # alone as evidence of the attempt; we do not synthesize an
        # assistant record. Anthropic's request-id on the failed
        # response is intentionally NOT captured on the user record.
        logger.info(
            "upstream returned error status=%d, conversation_id=%s",
            upstream_status,
            conversation_id,
        )

    response_headers = _filter_response_headers(upstream_response.headers)
    # x-llmdr-audited marks "this response was forwarded by llmdr",
    # which is true for any upstream response we deliver to the
    # client, success or upstream-error. Proxy-originated errors
    # (audit failure, streaming rejection, 404) do not set this
    # header because the response did not come from the upstream.
    response_headers[AUDITED_RESPONSE_HEADER] = "true"

    return Response(
        content=upstream_body,
        status_code=upstream_status,
        headers=response_headers,
        media_type=upstream_response.headers.get("content-type"),
    )


def _write_audit_record(
    *,
    config: Config,
    conversation_id: str,
    request_id: str | None,
    role: Literal["user", "assistant"],
    content: str,
    model: str,
) -> None:
    """Open a fresh AuditStore on the calling thread, write, and close.

    Opening per call is deliberate. ``sqlite3.Connection`` defaults to
    ``check_same_thread=True``, and the proxy dispatches audit writes
    to asyncio's default executor, which does not pin a request to one
    worker thread across the two writes (user pre-call, assistant
    post-call). Opening, writing, and closing inside the same
    :func:`asyncio.to_thread` call guarantees the connection is only
    ever touched by the thread that created it. SQLite WAL handles the
    short-lived concurrent connections cleanly, and local-disk open is
    sub-millisecond, so the per-write open is cheap.
    """
    with AuditStore(config.db_path, config.encryption_key) as store:
        store.write_record(
            source="live",
            conversation_id=conversation_id,
            request_id=request_id,
            role=role,
            content=content,
            model=model,
            metadata={},
            detection_status="skipped",
        )


def _resolve_conversation_id(headers: Mapping[str, str]) -> str | None:
    """Return a conversation id for the current request.

    If the client supplied ``x-llmdr-conversation-id`` it is used after
    a length check. Otherwise a fresh UUID4 is generated. Returns
    ``None`` only when the client header was present but failed
    validation, which the caller surfaces as a 400.
    """
    client_value = headers.get(CONVERSATION_ID_HEADER)
    if client_value is None:
        return str(uuid.uuid4())
    if not client_value or len(client_value) > _MAX_CONVERSATION_ID_LEN:
        return None
    return client_value


def _filter_request_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Build the header dict forwarded to upstream.

    Strips RFC 7230 hop-by-hop headers and any proxy-only headers
    consumed by llmdr. The client's authentication header
    (``x-api-key``) and Anthropic protocol headers (``anthropic-version``,
    ``anthropic-beta``) pass through unchanged.
    """
    filtered: dict[str, str] = {}
    for name, value in headers.items():
        lower = name.lower()
        if lower in _HOP_BY_HOP_HEADERS:
            continue
        if lower in _PROXY_ONLY_REQUEST_HEADERS:
            continue
        filtered[name] = value
    return filtered


def _filter_response_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Build the header dict returned to the client.

    Strips RFC 7230 hop-by-hop headers. The remaining headers
    (including ``content-type``, ``request-id``, and the
    ``anthropic-ratelimit-*`` family) are forwarded unchanged.
    """
    filtered: dict[str, str] = {}
    for name, value in headers.items():
        if name.lower() in _HOP_BY_HOP_HEADERS:
            continue
        filtered[name] = value
    return filtered


def _error_response(
    *, status_code: int, error_type: str, message: str
) -> JSONResponse:
    """Construct a JSON error response in the llmdr error envelope.

    The envelope shape mirrors Anthropic's ``{"error": {"type", "message"}}``
    so existing error-handling code on the client side continues to
    work; the ``type`` value carries an ``llmdr_*`` prefix so it is
    unambiguously a proxy-originated error rather than an upstream one.
    """
    return JSONResponse(
        status_code=status_code,
        content={"error": {"type": error_type, "message": message}},
    )
