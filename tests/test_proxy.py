"""Tests for the FastAPI proxy.

Organized by what each test proves rather than which method it exercises.
One TestClass per concern:

* TestAppConstruction: create_app produces the expected FastAPI surface.
* TestRoutingAndUnsupported: only POST /v1/messages is handled; everything
  else returns an llmdr-shaped 404.
* TestRequestValidation: client-side input is rejected before any audit
  or upstream call.
* TestConversationId: per-request UUID by default, optional client header
  override with length bounds; the header is consumed, not forwarded.
* TestSplitWriteTiming: user record written before upstream call,
  assistant record after; both records carry the right fields.
* TestRequestForwardingFidelity: body bytes, auth headers, and protocol
  headers reach upstream unchanged; hop-by-hop and proxy-only headers
  are stripped.
* TestResponseForwardingFidelity: upstream status and body are forwarded
  byte-for-byte; Anthropic response headers are preserved;
  x-llmdr-audited marks every forwarded response and is absent from
  proxy-originated errors.
* TestFailureModes: every failure path produces the right status and the
  right audit-store state; the chain remains intact under failure.
* TestLoggingDiscipline: no log record in this module references payload
  content. Structural enforcement of the no-content-logging rule.
* TestAuditStoreIntegration: sequential and concurrent requests both
  produce intact chains; the concurrent case is the regression guard
  for the sqlite3 check_same_thread bug.
* TestLifespan: the real lifespan starts and stops cleanly.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

import httpx
import pytest
from cryptography.fernet import Fernet
from fastapi import FastAPI
from fastapi.testclient import TestClient

import llmdr.proxy
from llmdr.audit import AuditRecord, AuditStore
from llmdr.config import Config
from llmdr.proxy import (
    AUDITED_RESPONSE_HEADER,
    CONVERSATION_ID_HEADER,
    MESSAGES_PATH,
    create_app,
)


# -----------------------------------------------------------------------------
# Fixtures and helpers
# -----------------------------------------------------------------------------


@pytest.fixture
def proxy_config(tmp_path: Path) -> Config:
    return Config(
        encryption_key=Fernet.generate_key(),
        db_path=tmp_path / "proxy.db",
        upstream_base_url="https://api.anthropic.com",
        proxy_host="127.0.0.1",
        proxy_port=8080,
        log_level="INFO",
        default_analyst_id=None,
    )


class MockUpstream:
    """Records every upstream request and returns canned responses.

    Tests ``queue_response`` or ``queue_exception`` for specific cases;
    when the queue is empty the default 200 is returned so tests that
    exercise many requests (concurrency, sequential chain checks) do
    not have to enumerate every reply.
    """

    DEFAULT_BODY = (
        b'{"id":"msg_default","type":"message",'
        b'"role":"assistant","model":"claude-default",'
        b'"content":[{"type":"text","text":"ok"}]}'
    )
    DEFAULT_HEADERS = {
        "content-type": "application/json",
        "request-id": "req-default",
    }

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self._queue: list[Any] = []

    def queue_response(
        self,
        *,
        status: int = 200,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        merged = dict(self.DEFAULT_HEADERS)
        if headers:
            merged.update(headers)
        self._queue.append(
            httpx.Response(
                status,
                content=self.DEFAULT_BODY if body is None else body,
                headers=merged,
            )
        )

    def queue_exception(self, exc: Exception) -> None:
        self._queue.append(exc)

    def _handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self._queue:
            entry = self._queue.pop(0)
            if isinstance(entry, Exception):
                raise entry
            return entry
        return httpx.Response(
            200,
            content=self.DEFAULT_BODY,
            headers=dict(self.DEFAULT_HEADERS),
        )

    def install(self, app: FastAPI, base_url: str) -> None:
        app.state.upstream = httpx.AsyncClient(
            transport=httpx.MockTransport(self._handler),
            base_url=base_url,
        )


@pytest.fixture
def mock_upstream() -> MockUpstream:
    return MockUpstream()


@pytest.fixture
def proxy_app(proxy_config: Config, mock_upstream: MockUpstream) -> FastAPI:
    app = create_app(proxy_config)
    mock_upstream.install(app, proxy_config.upstream_base_url)
    return app


@pytest.fixture
def client(proxy_app: FastAPI) -> TestClient:
    """A TestClient *without* the `with` context manager, intentionally.

    Entering TestClient as a context manager triggers the app's lifespan,
    which constructs a real httpx.AsyncClient and overwrites the
    MockUpstream installed by the proxy_app fixture. Tests in
    TestLifespan exercise the real lifespan with their own TestClient.
    """
    return TestClient(proxy_app)


@pytest.fixture
def read_audit_records(proxy_config: Config) -> Callable[[], list[AuditRecord]]:
    def _read() -> list[AuditRecord]:
        with AuditStore(proxy_config.db_path, proxy_config.encryption_key) as store:
            return store.get_records(
                analyst_id="test-analyst",
                justification="test read",
            )

    return _read


@pytest.fixture
def valid_request_body() -> dict[str, Any]:
    return {
        "model": "claude-opus-4-7",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "hello"}],
    }


# -----------------------------------------------------------------------------
# TestAppConstruction
# -----------------------------------------------------------------------------


class TestAppConstruction:
    """create_app returns a configured FastAPI with the expected surface."""

    def test_create_app_returns_fastapi_with_config_on_state(self, proxy_config):
        app = create_app(proxy_config)
        assert isinstance(app, FastAPI)
        assert app.state.config is proxy_config

    def test_routes_registered_for_messages_and_catchall(self, proxy_config):
        app = create_app(proxy_config)
        paths_methods = [
            (r.path, set(r.methods))
            for r in app.routes
            if hasattr(r, "methods") and hasattr(r, "path")
        ]
        assert (MESSAGES_PATH, {"POST"}) in paths_methods
        catchall = next(
            (m for p, m in paths_methods if p == "/{full_path:path}"),
            None,
        )
        assert catchall is not None
        assert catchall >= {
            "GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD",
        }

    def test_docs_and_openapi_endpoints_disabled(self, client):
        # /docs, /redoc, /openapi.json all fall through to the catch-all 404.
        for path in ("/docs", "/redoc", "/openapi.json"):
            response = client.get(path)
            assert response.status_code == 404
            assert response.json()["error"]["type"] == "llmdr_unsupported"


# -----------------------------------------------------------------------------
# TestRoutingAndUnsupported
# -----------------------------------------------------------------------------


class TestRoutingAndUnsupported:
    """Anything other than POST /v1/messages returns an llmdr-shaped 404."""

    def test_get_to_messages_path_returns_404_unsupported(self, client):
        response = client.get(MESSAGES_PATH)
        assert response.status_code == 404
        assert response.json()["error"]["type"] == "llmdr_unsupported"

    def test_post_to_other_path_returns_404_unsupported(self, client):
        response = client.post("/v1/other", json={})
        assert response.status_code == 404
        assert response.json()["error"]["type"] == "llmdr_unsupported"

    def test_root_returns_404_unsupported(self, client):
        response = client.get("/")
        assert response.status_code == 404
        assert response.json()["error"]["type"] == "llmdr_unsupported"

    def test_404_response_matches_error_envelope(self, client):
        response = client.get("/anything")
        body = response.json()
        assert set(body.keys()) == {"error"}
        assert set(body["error"].keys()) == {"type", "message"}

    def test_404_does_not_carry_audited_header(self, client):
        response = client.get("/anything")
        assert AUDITED_RESPONSE_HEADER not in response.headers


# -----------------------------------------------------------------------------
# TestRequestValidation
# -----------------------------------------------------------------------------


class TestRequestValidation:
    """Bad requests fail before any audit or upstream activity."""

    def test_empty_body_returns_400_bad_request(self, client, mock_upstream, read_audit_records):
        response = client.post(MESSAGES_PATH, content=b"")
        assert response.status_code == 400
        assert response.json()["error"]["type"] == "llmdr_bad_request"
        assert mock_upstream.requests == []
        assert read_audit_records() == []

    def test_non_utf8_body_returns_400_no_audit_record_no_upstream(
        self, client, mock_upstream, read_audit_records
    ):
        # 0xFF is invalid as a UTF-8 lead byte.
        response = client.post(
            MESSAGES_PATH,
            content=b"\xff\xff\xff\xff",
            headers={"content-type": "application/json"},
        )
        assert response.status_code == 400
        assert response.json()["error"]["type"] == "llmdr_bad_request"
        assert "UTF-8" in response.json()["error"]["message"]
        assert mock_upstream.requests == []
        assert read_audit_records() == []

    def test_invalid_json_returns_400(self, client, mock_upstream, read_audit_records):
        response = client.post(
            MESSAGES_PATH,
            content=b"not valid json {",
            headers={"content-type": "application/json"},
        )
        assert response.status_code == 400
        assert response.json()["error"]["type"] == "llmdr_bad_request"
        assert mock_upstream.requests == []
        assert read_audit_records() == []

    @pytest.mark.parametrize(
        "raw",
        [b"[]", b'"hello"', b"42", b"null", b"true"],
        ids=["list", "string", "number", "null", "bool"],
    )
    def test_json_non_object_body_returns_400(self, client, mock_upstream, raw):
        response = client.post(
            MESSAGES_PATH,
            content=raw,
            headers={"content-type": "application/json"},
        )
        assert response.status_code == 400
        assert response.json()["error"]["type"] == "llmdr_bad_request"
        assert mock_upstream.requests == []

    def test_streaming_request_returns_400_unsupported_does_not_hit_upstream(
        self, client, mock_upstream, read_audit_records, valid_request_body
    ):
        body = dict(valid_request_body, stream=True)
        response = client.post(MESSAGES_PATH, json=body)
        assert response.status_code == 400
        assert response.json()["error"]["type"] == "llmdr_unsupported"
        # User addition #5: a rejected streaming request must not hit
        # upstream. A leaked forwarding here would be a forwarding bug
        # AND a cost leak (tokens billed for an unaudited interaction).
        assert mock_upstream.requests == []
        assert read_audit_records() == []

    def test_stream_false_is_forwarded_normally(self, client, mock_upstream, valid_request_body):
        body = dict(valid_request_body, stream=False)
        response = client.post(MESSAGES_PATH, json=body)
        assert response.status_code == 200
        assert len(mock_upstream.requests) == 1

    def test_stream_zero_is_treated_as_not_streaming(self, client, mock_upstream, valid_request_body):
        # Anthropic's contract treats only stream=true as streaming;
        # falsy non-bool should be forwarded as-is.
        body = dict(valid_request_body, stream=0)
        response = client.post(MESSAGES_PATH, json=body)
        assert response.status_code == 200
        assert len(mock_upstream.requests) == 1

    def test_missing_model_returns_400(self, client, mock_upstream):
        response = client.post(MESSAGES_PATH, json={"messages": []})
        assert response.status_code == 400
        assert response.json()["error"]["type"] == "llmdr_bad_request"
        assert mock_upstream.requests == []

    def test_empty_model_returns_400(self, client, mock_upstream):
        response = client.post(MESSAGES_PATH, json={"model": "", "messages": []})
        assert response.status_code == 400
        assert response.json()["error"]["type"] == "llmdr_bad_request"
        assert mock_upstream.requests == []

    def test_non_string_model_returns_400(self, client, mock_upstream):
        response = client.post(MESSAGES_PATH, json={"model": 42, "messages": []})
        assert response.status_code == 400
        assert response.json()["error"]["type"] == "llmdr_bad_request"
        assert mock_upstream.requests == []


# -----------------------------------------------------------------------------
# TestConversationId
# -----------------------------------------------------------------------------


class TestConversationId:
    """conversation_id resolution and the proxy-only header contract."""

    def test_no_header_generates_uuid_used_in_both_records(
        self, client, mock_upstream, valid_request_body, read_audit_records
    ):
        response = client.post(MESSAGES_PATH, json=valid_request_body)
        assert response.status_code == 200

        records = read_audit_records()
        assert len(records) == 2
        ids = {r.conversation_id for r in records}
        assert len(ids) == 1
        only_id = next(iter(ids))
        # UUID4 string form is 36 chars including hyphens.
        assert len(only_id) == 36

    def test_client_header_overrides_uuid(
        self, client, mock_upstream, valid_request_body, read_audit_records
    ):
        response = client.post(
            MESSAGES_PATH,
            json=valid_request_body,
            headers={CONVERSATION_ID_HEADER: "conv-supplied-by-client"},
        )
        assert response.status_code == 200
        records = read_audit_records()
        for r in records:
            assert r.conversation_id == "conv-supplied-by-client"

    def test_empty_header_value_returns_400(self, client, mock_upstream, valid_request_body):
        response = client.post(
            MESSAGES_PATH,
            json=valid_request_body,
            headers={CONVERSATION_ID_HEADER: ""},
        )
        assert response.status_code == 400
        assert response.json()["error"]["type"] == "llmdr_bad_request"
        assert mock_upstream.requests == []

    def test_header_value_exceeding_max_length_returns_400(
        self, client, mock_upstream, valid_request_body
    ):
        response = client.post(
            MESSAGES_PATH,
            json=valid_request_body,
            headers={CONVERSATION_ID_HEADER: "x" * 257},
        )
        assert response.status_code == 400
        assert response.json()["error"]["type"] == "llmdr_bad_request"
        assert mock_upstream.requests == []

    def test_header_consumed_not_forwarded_to_upstream(
        self, client, mock_upstream, valid_request_body
    ):
        response = client.post(
            MESSAGES_PATH,
            json=valid_request_body,
            headers={CONVERSATION_ID_HEADER: "conv-secret"},
        )
        assert response.status_code == 200
        upstream_header_names = {k.lower() for k in mock_upstream.requests[0].headers.keys()}
        assert CONVERSATION_ID_HEADER not in upstream_header_names


# -----------------------------------------------------------------------------
# TestSplitWriteTiming
# -----------------------------------------------------------------------------


class TestSplitWriteTiming:
    """User record before upstream, assistant record after, with correct fields."""

    def test_successful_round_trip_writes_exactly_two_records(
        self, client, mock_upstream, valid_request_body, read_audit_records
    ):
        response = client.post(MESSAGES_PATH, json=valid_request_body)
        assert response.status_code == 200
        records = read_audit_records()
        assert len(records) == 2
        assert records[0].role == "user"
        assert records[1].role == "assistant"

    def test_user_record_is_written_before_upstream_call(
        self, proxy_app, client, mock_upstream, valid_request_body, monkeypatch
    ):
        """Pin the temporal contract: user write happens before upstream call,
        assistant write happens after upstream returns. This is what makes
        the prompt forensically recoverable even when upstream fails.
        """
        sequence: list[str] = []

        original = llmdr.proxy._write_audit_record

        def tracking_write(**kwargs):
            sequence.append(f"write:{kwargs['role']}")
            original(**kwargs)

        def tracking_handler(request):
            sequence.append("upstream")
            return httpx.Response(
                200,
                content=mock_upstream.DEFAULT_BODY,
                headers=dict(mock_upstream.DEFAULT_HEADERS),
            )

        monkeypatch.setattr(llmdr.proxy, "_write_audit_record", tracking_write)
        proxy_app.state.upstream = httpx.AsyncClient(
            transport=httpx.MockTransport(tracking_handler),
            base_url="https://api.anthropic.com",
        )

        response = client.post(MESSAGES_PATH, json=valid_request_body)
        assert response.status_code == 200
        assert sequence == ["write:user", "upstream", "write:assistant"]

    def test_user_record_request_id_is_none(
        self, client, mock_upstream, valid_request_body, read_audit_records
    ):
        # Even though the upstream response has a request-id, the user
        # record's column stays None per its documented default.
        mock_upstream.queue_response(headers={"request-id": "req-upstream-1"})
        client.post(MESSAGES_PATH, json=valid_request_body)
        records = read_audit_records()
        user_record = next(r for r in records if r.role == "user")
        assert user_record.request_id is None

    def test_assistant_record_request_id_comes_from_upstream_header(
        self, client, mock_upstream, valid_request_body, read_audit_records
    ):
        mock_upstream.queue_response(headers={"request-id": "req-upstream-77"})
        client.post(MESSAGES_PATH, json=valid_request_body)
        records = read_audit_records()
        assistant = next(r for r in records if r.role == "assistant")
        assert assistant.request_id == "req-upstream-77"

    def test_user_and_assistant_share_conversation_id(
        self, client, mock_upstream, valid_request_body, read_audit_records
    ):
        client.post(MESSAGES_PATH, json=valid_request_body)
        records = read_audit_records()
        assert records[0].conversation_id == records[1].conversation_id

    def test_user_record_content_preserves_raw_request_bytes_with_unicode_and_key_order(
        self, client, mock_upstream, read_audit_records
    ):
        # Deliberately non-alphabetical key order, unicode in values,
        # extra non-Anthropic fields. The proxy stores the decoded raw
        # body, not a re-serialized parse, so this should round-trip
        # byte-for-byte. Pins that the audit record is the actual
        # request the client sent, not a normalized version of it.
        body_str = (
            '{"model": "claude-test", '
            '"messages": [{"role": "user", "content": "hi"}], '
            '"z_field": "héllo 你好", '
            '"a_field": 1}'
        )
        body_bytes = body_str.encode("utf-8")
        response = client.post(
            MESSAGES_PATH,
            content=body_bytes,
            headers={"content-type": "application/json"},
        )
        assert response.status_code == 200
        records = read_audit_records()
        user_record = next(r for r in records if r.role == "user")
        # Content stored is the exact UTF-8 decode of the bytes sent,
        # preserving key order, whitespace, unicode, and extra fields.
        assert user_record.content == body_str

    def test_both_records_have_source_live_detection_skipped_metadata_empty(
        self, client, mock_upstream, valid_request_body, read_audit_records
    ):
        client.post(MESSAGES_PATH, json=valid_request_body)
        records = read_audit_records()
        for r in records:
            assert r.source == "live"
            assert r.detection_status == "skipped"
            assert r.metadata == {}

    def test_both_records_carry_request_body_model_field(
        self, client, mock_upstream, valid_request_body, read_audit_records
    ):
        # Even if the upstream response embeds a different model name
        # (snapshot vs alias), the audit records carry the model the
        # client requested.
        mock_upstream.queue_response(
            body=b'{"id":"m1","model":"claude-other-snapshot","content":[]}',
        )
        client.post(MESSAGES_PATH, json=valid_request_body)
        records = read_audit_records()
        for r in records:
            assert r.model == valid_request_body["model"]


# -----------------------------------------------------------------------------
# TestRequestForwardingFidelity
# -----------------------------------------------------------------------------


class TestRequestForwardingFidelity:
    """Bytes and headers reach upstream unchanged, except for stripped headers."""

    def test_request_body_bytes_forwarded_verbatim(
        self, client, mock_upstream
    ):
        raw = (
            b'{"model": "claude-x", '
            b'"messages": [{"role": "user", "content": "hi"}], '
            b'"z_field": "h\xc3\xa9llo \xe4\xbd\xa0\xe5\xa5\xbd", '
            b'"a_field": 1}'
        )
        response = client.post(
            MESSAGES_PATH,
            content=raw,
            headers={"content-type": "application/json"},
        )
        assert response.status_code == 200
        assert mock_upstream.requests[0].content == raw

    def test_x_api_key_header_forwarded(self, client, mock_upstream, valid_request_body):
        client.post(
            MESSAGES_PATH,
            json=valid_request_body,
            headers={"x-api-key": "sk-test-client-key"},
        )
        assert mock_upstream.requests[0].headers.get("x-api-key") == "sk-test-client-key"

    def test_anthropic_protocol_headers_forwarded(
        self, client, mock_upstream, valid_request_body
    ):
        client.post(
            MESSAGES_PATH,
            json=valid_request_body,
            headers={
                "anthropic-version": "2023-06-01",
                "anthropic-beta": "messages-2025-01-01",
            },
        )
        assert mock_upstream.requests[0].headers.get("anthropic-version") == "2023-06-01"
        assert (
            mock_upstream.requests[0].headers.get("anthropic-beta")
            == "messages-2025-01-01"
        )

    def test_hop_by_hop_request_headers_not_forwarded_as_client_value(
        self, client, mock_upstream, valid_request_body
    ):
        # The client's literal values for hop-by-hop headers must not
        # appear at upstream. httpx may set its own Connection or
        # Transfer-Encoding values; what matters is the client's
        # injected values do not propagate.
        client.post(
            MESSAGES_PATH,
            json=valid_request_body,
            headers={
                "Connection": "X-Client-Connection-Sentinel",
                "Keep-Alive": "X-Client-KeepAlive-Sentinel",
                "Upgrade": "X-Client-Upgrade-Sentinel",
                "Proxy-Authorization": "X-Client-Proxy-Sentinel",
            },
        )
        upstream_headers = mock_upstream.requests[0].headers
        for value in (
            "X-Client-Connection-Sentinel",
            "X-Client-KeepAlive-Sentinel",
            "X-Client-Upgrade-Sentinel",
            "X-Client-Proxy-Sentinel",
        ):
            assert value not in list(upstream_headers.values())

    def test_host_header_not_forwarded_as_client_value(
        self, client, mock_upstream, valid_request_body
    ):
        # TestClient sends host=testserver. httpx sets its own Host
        # based on the upstream base_url; the client's incoming Host
        # must not propagate.
        client.post(MESSAGES_PATH, json=valid_request_body)
        assert mock_upstream.requests[0].headers.get("host") != "testserver"

    def test_x_llmdr_conversation_id_consumed_not_forwarded(
        self, client, mock_upstream, valid_request_body
    ):
        client.post(
            MESSAGES_PATH,
            json=valid_request_body,
            headers={CONVERSATION_ID_HEADER: "conv-private"},
        )
        upstream_header_names = {
            k.lower() for k in mock_upstream.requests[0].headers.keys()
        }
        assert CONVERSATION_ID_HEADER not in upstream_header_names


# -----------------------------------------------------------------------------
# TestResponseForwardingFidelity
# -----------------------------------------------------------------------------


class TestResponseForwardingFidelity:
    """Upstream responses reach the client byte-for-byte with audited marker."""

    def test_2xx_body_forwarded_verbatim(
        self, client, mock_upstream, valid_request_body
    ):
        upstream_body = (
            b'{"id":"msg_specific","type":"message",'
            b'"content":[{"type":"text","text":"reply"}],"model":"x"}'
        )
        mock_upstream.queue_response(status=200, body=upstream_body)
        response = client.post(MESSAGES_PATH, json=valid_request_body)
        assert response.status_code == 200
        assert response.content == upstream_body

    def test_upstream_400_body_and_status_forwarded(
        self, client, mock_upstream, valid_request_body
    ):
        body = (
            b'{"type":"error","error":'
            b'{"type":"invalid_request_error","message":"bad input"}}'
        )
        mock_upstream.queue_response(status=400, body=body)
        response = client.post(MESSAGES_PATH, json=valid_request_body)
        assert response.status_code == 400
        assert response.content == body

    def test_upstream_500_body_and_status_forwarded(
        self, client, mock_upstream, valid_request_body
    ):
        body = b'{"type":"error","error":{"type":"overloaded_error"}}'
        mock_upstream.queue_response(status=529, body=body)
        response = client.post(MESSAGES_PATH, json=valid_request_body)
        assert response.status_code == 529
        assert response.content == body

    def test_upstream_request_id_header_forwarded_to_client(
        self, client, mock_upstream, valid_request_body
    ):
        mock_upstream.queue_response(headers={"request-id": "req-from-anthropic"})
        response = client.post(MESSAGES_PATH, json=valid_request_body)
        assert response.headers.get("request-id") == "req-from-anthropic"

    def test_anthropic_ratelimit_headers_forwarded(
        self, client, mock_upstream, valid_request_body
    ):
        mock_upstream.queue_response(
            headers={
                "anthropic-ratelimit-requests-remaining": "100",
                "anthropic-ratelimit-tokens-remaining": "50000",
            }
        )
        response = client.post(MESSAGES_PATH, json=valid_request_body)
        assert response.headers.get("anthropic-ratelimit-requests-remaining") == "100"
        assert response.headers.get("anthropic-ratelimit-tokens-remaining") == "50000"

    def test_audited_header_present_on_2xx_and_upstream_errors_absent_on_proxy_errors(
        self, client, mock_upstream, valid_request_body
    ):
        # 2xx forwarded.
        r = client.post(MESSAGES_PATH, json=valid_request_body)
        assert r.headers.get(AUDITED_RESPONSE_HEADER) == "true"

        # Upstream error forwarded.
        mock_upstream.queue_response(status=400, body=b"{}")
        r = client.post(MESSAGES_PATH, json=valid_request_body)
        assert r.headers.get(AUDITED_RESPONSE_HEADER) == "true"

        # Proxy-originated 400 (streaming) has no marker.
        r = client.post(
            MESSAGES_PATH, json=dict(valid_request_body, stream=True)
        )
        assert AUDITED_RESPONSE_HEADER not in r.headers


# -----------------------------------------------------------------------------
# TestFailureModes
# -----------------------------------------------------------------------------


class TestFailureModes:
    """Each failure path produces the right status and audit-store state."""

    def test_user_audit_write_failure_returns_500_skips_upstream_no_records(
        self,
        client,
        mock_upstream,
        valid_request_body,
        read_audit_records,
        monkeypatch,
    ):
        def failing(**kwargs):
            raise sqlite3.OperationalError("simulated")

        monkeypatch.setattr(llmdr.proxy, "_write_audit_record", failing)
        response = client.post(MESSAGES_PATH, json=valid_request_body)
        assert response.status_code == 500
        assert response.json()["error"]["type"] == "llmdr_audit_failure"
        assert mock_upstream.requests == []
        assert read_audit_records() == []
        assert AUDITED_RESPONSE_HEADER not in response.headers

    def test_assistant_audit_write_failure_returns_500_user_record_persists(
        self,
        client,
        mock_upstream,
        valid_request_body,
        read_audit_records,
        monkeypatch,
    ):
        original = llmdr.proxy._write_audit_record

        def selective(**kwargs):
            if kwargs.get("role") == "assistant":
                raise sqlite3.OperationalError("simulated")
            return original(**kwargs)

        monkeypatch.setattr(llmdr.proxy, "_write_audit_record", selective)
        mock_upstream.queue_response(status=200)
        response = client.post(MESSAGES_PATH, json=valid_request_body)
        assert response.status_code == 500
        assert response.json()["error"]["type"] == "llmdr_audit_failure"
        assert AUDITED_RESPONSE_HEADER not in response.headers
        records = read_audit_records()
        assert len(records) == 1
        assert records[0].role == "user"

    def test_upstream_connect_error_returns_502_user_record_persists(
        self, client, mock_upstream, valid_request_body, read_audit_records
    ):
        mock_upstream.queue_exception(httpx.ConnectError("simulated"))
        response = client.post(MESSAGES_PATH, json=valid_request_body)
        assert response.status_code == 502
        assert response.json()["error"]["type"] == "llmdr_upstream_unreachable"
        assert AUDITED_RESPONSE_HEADER not in response.headers
        records = read_audit_records()
        assert len(records) == 1
        assert records[0].role == "user"
        assert records[0].request_id is None

    def test_upstream_read_timeout_returns_502(
        self, client, mock_upstream, valid_request_body, read_audit_records
    ):
        mock_upstream.queue_exception(httpx.ReadTimeout("simulated"))
        response = client.post(MESSAGES_PATH, json=valid_request_body)
        assert response.status_code == 502
        assert response.json()["error"]["type"] == "llmdr_upstream_unreachable"
        records = read_audit_records()
        assert len(records) == 1

    def test_upstream_4xx_writes_user_record_only(
        self, client, mock_upstream, valid_request_body, read_audit_records
    ):
        mock_upstream.queue_response(status=400, body=b'{"error":"bad"}')
        response = client.post(MESSAGES_PATH, json=valid_request_body)
        assert response.status_code == 400
        records = read_audit_records()
        assert len(records) == 1
        assert records[0].role == "user"

    def test_upstream_5xx_writes_user_record_only(
        self, client, mock_upstream, valid_request_body, read_audit_records
    ):
        mock_upstream.queue_response(status=529, body=b'{"error":"overloaded"}')
        response = client.post(MESSAGES_PATH, json=valid_request_body)
        assert response.status_code == 529
        records = read_audit_records()
        assert len(records) == 1
        assert records[0].role == "user"

    def test_user_record_request_id_stays_none_when_upstream_returns_error_with_request_id_header(
        self, client, mock_upstream, valid_request_body, read_audit_records
    ):
        # Clarification 1 invariant: even when Anthropic returns a
        # request-id on a 4xx, that id is NOT captured on the user
        # record. The user record represents the attempt, not the
        # failed response.
        mock_upstream.queue_response(
            status=400,
            body=b'{"type":"error","error":{"type":"invalid_request_error"}}',
            headers={"request-id": "req-from-failed-upstream"},
        )
        response = client.post(MESSAGES_PATH, json=valid_request_body)
        assert response.status_code == 400
        records = read_audit_records()
        assert len(records) == 1
        assert records[0].role == "user"
        assert records[0].request_id is None

    def test_non_utf8_upstream_body_on_2xx_returns_500_user_record_persists(
        self, client, mock_upstream, valid_request_body, read_audit_records
    ):
        mock_upstream.queue_response(status=200, body=b"\xff\xfe\x00\x01\x02")
        response = client.post(MESSAGES_PATH, json=valid_request_body)
        assert response.status_code == 500
        assert response.json()["error"]["type"] == "llmdr_upstream_invalid"
        records = read_audit_records()
        assert len(records) == 1
        assert records[0].role == "user"

    def test_chain_verifies_clean_after_mixed_failure_modes(
        self, proxy_config, client, mock_upstream, valid_request_body
    ):
        # Three requests covering: success, upstream 4xx, upstream
        # connection error. After all three the chain must still
        # verify clean (no half-written records, no orphan rows).
        mock_upstream.queue_response(status=200)
        mock_upstream.queue_response(status=400, body=b'{"err":1}')
        mock_upstream.queue_exception(httpx.ConnectError("simulated"))

        client.post(MESSAGES_PATH, json=valid_request_body)
        client.post(MESSAGES_PATH, json=valid_request_body)
        client.post(MESSAGES_PATH, json=valid_request_body)

        with AuditStore(proxy_config.db_path, proxy_config.encryption_key) as store:
            result = store.verify_chain(analyst_id="test")
            assert result.ok is True

    def test_audit_failure_response_has_error_envelope_not_response_body(
        self,
        client,
        mock_upstream,
        valid_request_body,
        monkeypatch,
    ):
        # When the assistant audit write fails after upstream success,
        # the client gets the llmdr error envelope, not the upstream
        # response body. Forensic contract: an unaudited response is
        # never delivered, and "never delivered" includes "no partial
        # body bleed into the error envelope."
        upstream_body = b'{"id":"msg_secret","content":[{"type":"text","text":"private"}]}'
        mock_upstream.queue_response(status=200, body=upstream_body)
        original = llmdr.proxy._write_audit_record

        def selective(**kwargs):
            if kwargs.get("role") == "assistant":
                raise sqlite3.OperationalError("simulated")
            return original(**kwargs)

        monkeypatch.setattr(llmdr.proxy, "_write_audit_record", selective)
        response = client.post(MESSAGES_PATH, json=valid_request_body)
        assert response.status_code == 500
        assert response.json()["error"]["type"] == "llmdr_audit_failure"
        assert b"msg_secret" not in response.content
        assert b"private" not in response.content


# -----------------------------------------------------------------------------
# TestLoggingDiscipline
# -----------------------------------------------------------------------------


def _record_carries_sentinel(record: logging.LogRecord, sentinel: str) -> bool:
    if sentinel in record.getMessage():
        return True
    if isinstance(record.args, tuple):
        for arg in record.args:
            if sentinel in str(arg):
                return True
    elif record.args is not None:
        if sentinel in str(record.args):
            return True
    if record.exc_text and sentinel in record.exc_text:
        return True
    return False


class TestLoggingDiscipline:
    """No log record emitted by proxy.py references payload content.

    This is the structural enforcement of clarification 2: payload bytes
    live only in the encrypted audit store. If a future log statement
    in proxy.py references body content, these tests fail.
    """

    def test_no_log_record_contains_request_body_sentinel_on_success(
        self, client, mock_upstream, caplog
    ):
        sentinel = "SENTINEL_REQUEST_TOKEN_a1b2c3d4e5"
        body = {
            "model": "claude-test",
            "messages": [{"role": "user", "content": sentinel}],
        }
        with caplog.at_level(logging.DEBUG, logger="llmdr.proxy"):
            response = client.post(MESSAGES_PATH, json=body)
        assert response.status_code == 200
        for record in caplog.records:
            assert not _record_carries_sentinel(record, sentinel), (
                f"sentinel leaked into log record: {record.name} "
                f"{record.levelname} {record.getMessage()!r}"
            )

    def test_no_log_record_contains_response_body_sentinel_on_success(
        self, client, mock_upstream, valid_request_body, caplog
    ):
        sentinel = "SENTINEL_RESPONSE_TOKEN_f6g7h8i9j0"
        body = json.dumps(
            {"id": "m1", "content": [{"type": "text", "text": sentinel}]}
        ).encode("utf-8")
        mock_upstream.queue_response(status=200, body=body)
        with caplog.at_level(logging.DEBUG, logger="llmdr.proxy"):
            response = client.post(MESSAGES_PATH, json=valid_request_body)
        assert response.status_code == 200
        for record in caplog.records:
            assert not _record_carries_sentinel(record, sentinel), (
                f"sentinel leaked into log record: {record.getMessage()!r}"
            )

    def test_no_log_record_contains_upstream_error_body_sentinel(
        self, client, mock_upstream, valid_request_body, caplog
    ):
        sentinel = "SENTINEL_ERROR_BODY_k1l2m3n4o5"
        body = json.dumps(
            {"type": "error", "error": {"type": "invalid_request_error", "message": sentinel}}
        ).encode("utf-8")
        mock_upstream.queue_response(status=400, body=body)
        with caplog.at_level(logging.DEBUG, logger="llmdr.proxy"):
            response = client.post(MESSAGES_PATH, json=valid_request_body)
        assert response.status_code == 400
        for record in caplog.records:
            assert not _record_carries_sentinel(record, sentinel), (
                f"sentinel leaked into log record: {record.getMessage()!r}"
            )


# -----------------------------------------------------------------------------
# TestAuditStoreIntegration
# -----------------------------------------------------------------------------


class TestAuditStoreIntegration:
    """The proxy + AuditStore composition produces intact chains."""

    def test_three_sequential_requests_produce_six_records_with_intact_chain(
        self, proxy_config, client, mock_upstream, valid_request_body
    ):
        for _ in range(3):
            response = client.post(MESSAGES_PATH, json=valid_request_body)
            assert response.status_code == 200

        with AuditStore(proxy_config.db_path, proxy_config.encryption_key) as store:
            records = store.get_records(
                analyst_id="test", justification="test"
            )
            assert len(records) == 6
            assert [r.role for r in records] == ["user", "assistant"] * 3
            result = store.verify_chain(analyst_id="test")
            assert result.ok is True

    def test_concurrent_requests_succeed_and_chain_remains_intact(
        self, proxy_config, client, mock_upstream, valid_request_body
    ):
        """Regression guard for the sqlite3 check_same_thread bug.

        The original Depends-based per-request AuditStore would
        intermittently fail under concurrent load: a connection opened
        on one worker thread (the FastAPI dependency) would be reused
        from a different worker thread (asyncio.to_thread), raising
        sqlite3.ProgrammingError. The per-write open design fixed that.

        Ten concurrent requests through ten threads forces audit writes
        across many different executor workers; if a future refactor
        reintroduces per-request open without same-thread guarantees,
        this test fails.
        """
        n = 10

        def one_request(_):
            return client.post(MESSAGES_PATH, json=valid_request_body)

        with ThreadPoolExecutor(max_workers=n) as ex:
            results = list(ex.map(one_request, range(n)))

        assert len(results) == n
        for r in results:
            assert r.status_code == 200, r.content

        with AuditStore(proxy_config.db_path, proxy_config.encryption_key) as store:
            records = store.get_records(
                analyst_id="test", justification="test"
            )
            assert len(records) == n * 2
            roles = sorted(r.role for r in records)
            assert roles == ["assistant"] * n + ["user"] * n
            result = store.verify_chain(analyst_id="test")
            assert result.ok is True

    def test_per_write_open_uses_same_physical_db_file(
        self, proxy_config, client, mock_upstream, valid_request_body
    ):
        # After multiple requests the records all live in the configured
        # db_path. Pins that each per-write AuditStore open lands on the
        # same file (config-driven), not a per-process or per-thread
        # path artifact.
        for _ in range(2):
            client.post(MESSAGES_PATH, json=valid_request_body)
        conn = sqlite3.connect(str(proxy_config.db_path))
        try:
            row = conn.execute(
                "SELECT COUNT(*) FROM audit_records"
            ).fetchone()
            assert row[0] == 4
        finally:
            conn.close()

    def test_detection_status_and_metadata_round_trip_through_encryption(
        self, client, mock_upstream, valid_request_body, read_audit_records
    ):
        client.post(MESSAGES_PATH, json=valid_request_body)
        records = read_audit_records()
        assert len(records) == 2
        for r in records:
            assert r.detection_status == "skipped"
            assert r.metadata == {}


# -----------------------------------------------------------------------------
# TestLifespan
# -----------------------------------------------------------------------------


class TestLifespan:
    """The real lifespan starts and stops cleanly with a real httpx client."""

    def test_lifespan_enters_and_exits_cleanly_with_real_upstream_client(
        self, proxy_config
    ):
        # Use a fresh app *without* mocking upstream. The lifespan
        # constructs a real httpx.AsyncClient, we observe it on
        # app.state.upstream during the window, then exit cleanly.
        # No request is issued, so no network call is made.
        app = create_app(proxy_config)
        with TestClient(app) as test_client:
            assert isinstance(test_client.app.state.upstream, httpx.AsyncClient)
        # If aclose raised the with-exit would have propagated; reaching
        # this line is the test.
