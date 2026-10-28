"""Tests for HTML report generation.

Organized by what each test proves:

* TestReportBasics: a report generates for empty and populated stores
  and is a complete self-contained HTML document.
* TestChainBadge: the verification badge reflects real chain state,
  including the break location after tampering.
* TestAccessLogging: report generation is itself recorded; exactly two
  access_log entries per report (verify + read), carrying the analyst
  and justification.
* TestContentPolicy: content is absent by default, present only with
  include_content=True, and always HTML-escaped.
* TestHighlighting: PII spans from stored hits are wrapped in <mark>;
  span merging and malformed spans are handled.
* TestStatsAndTimeline: summary numbers and per-record rows reflect
  the stored records and their detection metadata.
* TestFilters: last and conversation_id selections narrow the report.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest
from cryptography.fernet import Fernet

from llmdr.audit import AuditStore
from llmdr.report import generate_report


@pytest.fixture
def fernet_key() -> bytes:
    return Fernet.generate_key()


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "report.db"


@pytest.fixture
def store(db_path: Path, fernet_key: bytes):
    with AuditStore(db_path, fernet_key) as s:
        yield s


def _write(
    store: AuditStore,
    *,
    conv: str = "conv-1",
    role: str = "user",
    content: str = "hello there",
    model: str = "claude-test",
    metadata: dict[str, Any] | None = None,
    detection_status: str = "ran_clean",
) -> int:
    return store.write_record(
        source="live",
        conversation_id=conv,
        request_id=None,
        role=role,  # type: ignore[arg-type]
        content=content,
        model=model,
        metadata=metadata if metadata is not None else {},
        detection_status=detection_status,  # type: ignore[arg-type]
    )


def _report(store: AuditStore, **kwargs: Any) -> str:
    kwargs.setdefault("analyst_id", "alice")
    kwargs.setdefault("justification", "test report")
    return generate_report(store, **kwargs).html


_PII_METADATA = {
    "detection": {
        "detectors_run": ["detect_secrets", "presidio"],
        "detectors_skipped": [],
        "errors": [],
        "hit_count": 1,
        "hits": [
            {
                "detector": "presidio",
                "type": "EMAIL_ADDRESS",
                "start": 12,
                "end": 32,
                "score": 1.0,
            }
        ],
    }
}
# Content aligned with the span above: chars 12..32 are the email.
_PII_CONTENT = "Reach me at jane.doe@example.com now"


# -----------------------------------------------------------------------------
# Basics
# -----------------------------------------------------------------------------


class TestReportBasics:
    def test_empty_store_generates_complete_document(self, store) -> None:
        html = _report(store)
        assert html.startswith("<!DOCTYPE html>")
        assert html.rstrip().endswith("</html>")
        assert "No records matched" in html
        assert "CHAIN INTACT" in html

    def test_populated_store_lists_records(self, store) -> None:
        _write(store, role="user", content="prompt one")
        _write(store, role="assistant", content="reply one")
        html = _report(store)
        assert "conv-1" in html
        assert "claude-test" in html

    def test_report_is_self_contained(self, store) -> None:
        _write(store)
        html = _report(store)
        # No external fetches: a forensic artifact must render offline
        # and must not phone home when opened.
        assert "http://" not in html
        assert "https://" not in html
        assert "<script" not in html

    def test_analyst_and_justification_in_header(self, store) -> None:
        html = _report(
            store,
            analyst_id="carol",
            justification="case 42 review",
            case_ref="CASE-42",
        )
        assert "carol" in html
        assert "case 42 review" in html
        assert "CASE-42" in html


# -----------------------------------------------------------------------------
# Chain badge
# -----------------------------------------------------------------------------


class TestChainBadge:
    def test_intact_chain_shows_intact_badge(self, store) -> None:
        _write(store)
        html = _report(store)
        assert "CHAIN INTACT" in html
        assert "CHAIN BROKEN" not in html

    def test_tampered_chain_shows_broken_badge_with_location(
        self, store, db_path, fernet_key
    ) -> None:
        _write(store, content="original one")
        _write(store, content="original two")
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute("UPDATE audit_records SET model = 'forged-model' WHERE id = 1")
            conn.commit()
        finally:
            conn.close()
        html = _report(store)
        assert "CHAIN BROKEN" in html
        assert "audit_records broken at id=1" in html
        assert "CHAIN INTACT" not in html


# -----------------------------------------------------------------------------
# Access logging
# -----------------------------------------------------------------------------


class TestAccessLogging:
    def test_report_writes_exactly_two_access_entries(self, store, db_path) -> None:
        _write(store)
        before = self._access_rows(db_path)
        _report(store, analyst_id="dave", justification="quarterly audit")
        rows = self._access_rows(db_path)
        assert len(rows) == len(before) + 2
        new_rows = rows[len(before) :]
        for analyst, justification in new_rows:
            assert analyst == "dave"
            assert "report generation" in justification
            assert "quarterly audit" in justification

    @staticmethod
    def _access_rows(db_path: Path) -> list[tuple[str, str]]:
        conn = sqlite3.connect(str(db_path))
        try:
            return conn.execute(
                "SELECT analyst_id, justification FROM access_log ORDER BY id"
            ).fetchall()
        finally:
            conn.close()

    def test_empty_analyst_rejected(self, store) -> None:
        with pytest.raises(ValueError):
            generate_report(store, analyst_id="", justification="x")


# -----------------------------------------------------------------------------
# Content policy
# -----------------------------------------------------------------------------


class TestContentPolicy:
    SECRET_CONTENT = "the confidential launch plan for project aurora"

    def test_content_absent_by_default(self, store) -> None:
        _write(store, content=self.SECRET_CONTENT)
        html = _report(store)
        assert self.SECRET_CONTENT not in html
        assert "omitted" in html

    def test_content_present_with_opt_in(self, store) -> None:
        _write(store, content=self.SECRET_CONTENT)
        html = _report(store, include_content=True)
        assert self.SECRET_CONTENT in html
        # The opt-in also surfaces the handling notice.
        assert "decrypted message content" in html

    def test_content_is_html_escaped(self, store) -> None:
        hostile = "<script>alert('xss')</script>"
        _write(store, content=hostile)
        html = _report(store, include_content=True)
        assert "<script>alert" not in html
        assert "&lt;script&gt;" in html

    def test_metadata_fields_are_html_escaped(self, store) -> None:
        _write(store, conv="<img src=x onerror=alert(1)>")
        html = _report(store)
        assert "<img src=x" not in html


# -----------------------------------------------------------------------------
# Highlighting
# -----------------------------------------------------------------------------


class TestHighlighting:
    def test_pii_span_wrapped_in_mark(self, store) -> None:
        _write(
            store,
            content=_PII_CONTENT,
            metadata=_PII_METADATA,
            detection_status="ran_with_hits",
        )
        html = _report(store, include_content=True)
        assert "<mark>jane.doe@example.com</mark>" in html

    def test_no_mark_without_content(self, store) -> None:
        _write(
            store,
            content=_PII_CONTENT,
            metadata=_PII_METADATA,
            detection_status="ran_with_hits",
        )
        html = _report(store)
        assert "<mark>" not in html

    def test_overlapping_spans_merge(self, store) -> None:
        metadata = {
            "detection": {
                "detectors_run": ["presidio"],
                "detectors_skipped": [],
                "errors": [],
                "hit_count": 2,
                "hits": [
                    {"detector": "presidio", "type": "A", "start": 0, "end": 5},
                    {"detector": "presidio", "type": "B", "start": 3, "end": 8},
                ],
            }
        }
        _write(
            store,
            content="0123456789",
            metadata=metadata,
            detection_status="ran_with_hits",
        )
        html = _report(store, include_content=True)
        assert "<mark>01234567</mark>" in html

    def test_malformed_spans_do_not_break_report(self, store) -> None:
        metadata = {
            "detection": {
                "detectors_run": ["presidio"],
                "detectors_skipped": [],
                "errors": [],
                "hit_count": 2,
                "hits": [
                    {"detector": "presidio", "type": "A", "start": 50, "end": 99},
                    {"detector": "presidio", "type": "B", "start": 5, "end": 2},
                ],
            }
        }
        _write(
            store,
            content="short",
            metadata=metadata,
            detection_status="ran_with_hits",
        )
        html = _report(store, include_content=True)
        assert "short" in html
        assert "<mark>" not in html


# -----------------------------------------------------------------------------
# Stats and timeline
# -----------------------------------------------------------------------------


class TestStatsAndTimeline:
    def test_hit_types_aggregated(self, store) -> None:
        _write(
            store,
            content=_PII_CONTENT,
            metadata=_PII_METADATA,
            detection_status="ran_with_hits",
        )
        _write(store, content="clean", detection_status="ran_clean")
        html = _report(store)
        assert "presidio: EMAIL_ADDRESS" in html
        assert "ran_with_hits" in html

    def test_detection_errors_surface(self, store) -> None:
        metadata = {
            "detection": {
                "detectors_run": [],
                "detectors_skipped": [],
                "errors": ["detect_secrets:RuntimeError"],
                "hit_count": 0,
                "hits": [],
            }
        }
        _write(store, metadata=metadata, detection_status="failed")
        html = _report(store)
        assert "detect_secrets:RuntimeError" in html

    def test_truncated_hit_list_flagged(self, store) -> None:
        metadata = {
            "detection": {
                "detectors_run": ["detect_secrets"],
                "detectors_skipped": [],
                "errors": [],
                "hit_count": 500,
                "hits": [
                    {"detector": "detect_secrets", "type": "Hex High Entropy String", "line": 1}
                ],
                "hits_truncated": True,
            }
        }
        _write(store, metadata=metadata, detection_status="ran_with_hits")
        html = _report(store)
        assert "hit list truncated" in html


# -----------------------------------------------------------------------------
# Filters
# -----------------------------------------------------------------------------


class TestFilters:
    def test_conversation_filter_narrows_report(self, store) -> None:
        _write(store, conv="conv-A", content="alpha content")
        _write(store, conv="conv-B", content="beta content")
        html = _report(store, conversation_id="conv-A")
        assert "conv-A" in html
        assert "conv-B" not in html
        assert "conversation_id=conv-A" in html

    def test_last_filter_limits_rows(self, store) -> None:
        for i in range(5):
            _write(store, conv=f"conv-{i}")
        html = _report(store, last=2)
        assert "conv-4" in html
        assert "conv-0" not in html
