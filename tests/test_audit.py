"""Tests for the AuditStore forensic core.

Organized by what each test proves rather than by which method it
exercises. One TestClass per forensic property:

* TestEmptyStore: a freshly initialized store is internally consistent.
* TestSingleRecord: the first write anchors to genesis and round-trips.
* TestMultiRecordChain: hashes link across writes; verification is a
  pure function of persisted state; conversation_id filtering works.
* TestAuditTamper: every audit_records mutation mode is caught and
  attributed to the exact row.
* TestAccessLogTamper: access_log mutation is attributed to the right
  chain, not the audit chain.
* TestAccessLogOnReadPaths: every read path produces an access_log
  entry with the right fields in the right columns.
* TestEncryption: plaintext round-trips; raw bytes on disk do not
  contain the plaintext.
* TestWriteRecordValidation: invalid input is rejected before any
  state change.
* TestGetRecordsValidation: read-path validation rejects before
  logging; rejected calls produce no access_log entry.
* TestDecryptionError: corrupted ciphertext is caught loudly and the
  access_log entry carries only the exception type name.
* TestCiphertextProperty: chain integrity is independent of the key.
* TestAtomicity: a transaction failure mid-write leaves no partial
  state and does not consume an id.
* TestKeyVersion: the configured key_version is persisted and surfaces
  on read.
"""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pytest
from cryptography.fernet import Fernet

from llmdr.audit import (
    GENESIS_HASH,
    AuditStore,
    DecryptionError,
)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def fernet_key() -> bytes:
    return Fernet.generate_key()


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "audit.db"


@pytest.fixture
def store(db_path: Path, fernet_key: bytes):
    with AuditStore(db_path, fernet_key) as s:
        yield s


@pytest.fixture
def valid_write_kwargs() -> dict[str, Any]:
    return {
        "source": "live",
        "conversation_id": "conv-1",
        "request_id": "req-1",
        "role": "user",
        "content": "hello world",
        "model": "claude-opus-4-7",
        "metadata": {"detector_hits": []},
        "detection_status": "ran_clean",
    }


@pytest.fixture
def analyst_ctx() -> dict[str, str]:
    return {"analyst_id": "analyst-1", "justification": "testing"}


@pytest.fixture
def raw_read(db_path: Path) -> Callable[..., list[tuple]]:
    """Read rows directly from the db without touching AuditStore.

    Opens and closes its own connection so it cannot interfere with the
    AuditStore's transaction state.
    """

    def _read(sql: str, *params: Any) -> list[tuple]:
        conn = sqlite3.connect(str(db_path))
        try:
            return conn.execute(sql, params).fetchall()
        finally:
            conn.close()

    return _read


@pytest.fixture
def tamper(db_path: Path) -> Callable[[str, int, str, Any], None]:
    """Mutate one column on one row, bypassing AuditStore.

    Simulates adversarial modification of the SQLite file. Always opens
    and closes its own connection.
    """

    def _tamper(table: str, row_id: int, column: str, new_value: Any) -> None:
        assert table in {"audit_records", "access_log"}, f"unknown table {table!r}"
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute(
                f"UPDATE {table} SET {column} = ? WHERE id = ?",  # noqa: S608
                (new_value, row_id),
            )
            conn.commit()
        finally:
            conn.close()

    return _tamper


# -----------------------------------------------------------------------------
# Atomicity test helpers
# -----------------------------------------------------------------------------


class _FailingInsertCursor:
    """Cursor wrapper that raises on INSERT INTO audit_records.

    Simulates a transaction failure after BEGIN IMMEDIATE but before the
    INSERT completes. BEGIN, SELECT, COMMIT, and ROLLBACK all pass
    through, so the AuditStore rollback path runs as it would in
    production.
    """

    def __init__(self, real_cursor: sqlite3.Cursor) -> None:
        self._cur = real_cursor

    def execute(self, sql: str, *args: Any, **kwargs: Any) -> "_FailingInsertCursor":
        if "INSERT INTO audit_records" in sql:
            raise sqlite3.OperationalError("simulated INSERT failure")
        self._cur.execute(sql, *args, **kwargs)
        return self

    def fetchone(self) -> Any:
        return self._cur.fetchone()

    def fetchall(self) -> list[Any]:
        return self._cur.fetchall()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cur, name)


class _FailingInsertConn:
    """Connection wrapper that hands out failing cursors.

    Used to monkey-patch AuditStore._conn for a single test. Every
    attribute other than cursor() proxies to the real connection.
    """

    def __init__(self, real_conn: sqlite3.Connection) -> None:
        self._conn = real_conn

    def cursor(self) -> _FailingInsertCursor:
        return _FailingInsertCursor(self._conn.cursor())

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


class TestEmptyStore:
    """A freshly initialized store is internally consistent, and access
    logging fires even when there is nothing to return."""

    def test_verify_chain_on_empty_store_is_ok(self, store, analyst_ctx):
        result = store.verify_chain(analyst_id=analyst_ctx["analyst_id"])
        assert result.ok is True
        assert result.audit_tampered_at is None
        assert result.access_log_tampered_at is None

    def test_get_records_on_empty_store_returns_empty(self, store, analyst_ctx):
        records = store.get_records(**analyst_ctx)
        assert records == []

    def test_get_records_on_empty_store_still_writes_access_log(
        self, store, analyst_ctx, raw_read
    ):
        store.get_records(**analyst_ctx)
        rows = raw_read(
            "SELECT row_count_returned, success, record_ids_accessed FROM access_log"
        )
        assert len(rows) == 1
        row_count, success, ids_json = rows[0]
        assert row_count == 0
        assert success == 1
        assert json.loads(ids_json) == []

    def test_verify_chain_writes_access_log_entry(self, store, analyst_ctx, raw_read):
        store.verify_chain(analyst_id=analyst_ctx["analyst_id"])
        rows = raw_read("SELECT analyst_id, success FROM access_log")
        assert len(rows) == 1
        assert rows[0][0] == analyst_ctx["analyst_id"]
        assert rows[0][1] == 1


class TestSingleRecord:
    """The first record anchors to the genesis sentinel and what comes
    back via get_records matches what went in."""

    def test_first_record_id_and_genesis_anchor(
        self, store, valid_write_kwargs, raw_read
    ):
        record_id = store.write_record(**valid_write_kwargs)
        assert record_id == 1
        rows = raw_read("SELECT prev_hash, row_hash FROM audit_records WHERE id = 1")
        prev_hash, row_hash = rows[0]
        assert prev_hash == GENESIS_HASH
        assert re.fullmatch(r"[0-9a-f]{64}", row_hash), (
            f"row_hash is not 64 lowercase hex chars: {row_hash!r}"
        )

    def test_round_trip_fields_match(self, store, valid_write_kwargs, analyst_ctx):
        store.write_record(**valid_write_kwargs)
        records = store.get_records(**analyst_ctx)
        assert len(records) == 1
        r = records[0]
        assert r.source == valid_write_kwargs["source"]
        assert r.conversation_id == valid_write_kwargs["conversation_id"]
        assert r.request_id == valid_write_kwargs["request_id"]
        assert r.role == valid_write_kwargs["role"]
        assert r.content == valid_write_kwargs["content"]
        assert r.model == valid_write_kwargs["model"]
        assert r.metadata == valid_write_kwargs["metadata"]
        assert r.detection_status == valid_write_kwargs["detection_status"]

    def test_verify_ok_after_single_write(self, store, valid_write_kwargs, analyst_ctx):
        store.write_record(**valid_write_kwargs)
        result = store.verify_chain(analyst_id=analyst_ctx["analyst_id"])
        assert result.ok is True


class TestMultiRecordChain:
    """Each row's prev_hash equals the previous row's row_hash, the
    first row anchors to genesis, and verification is purely a function
    of persisted state."""

    def test_chain_links_correctly_across_records(
        self, store, valid_write_kwargs, raw_read
    ):
        for i in range(5):
            kwargs = {
                **valid_write_kwargs,
                "conversation_id": f"conv-{i % 2}",
                "content": f"message {i}",
            }
            store.write_record(**kwargs)
        rows = raw_read(
            "SELECT id, prev_hash, row_hash FROM audit_records ORDER BY id ASC"
        )
        assert [r[0] for r in rows] == [1, 2, 3, 4, 5]
        assert rows[0][1] == GENESIS_HASH
        for i in range(1, 5):
            assert rows[i][1] == rows[i - 1][2], (
                f"row {rows[i][0]} prev_hash does not match row "
                f"{rows[i - 1][0]} row_hash"
            )

    def test_verify_ok_after_five_writes(self, store, valid_write_kwargs, analyst_ctx):
        for i in range(5):
            store.write_record(**{**valid_write_kwargs, "content": f"msg {i}"})
        result = store.verify_chain(analyst_id=analyst_ctx["analyst_id"])
        assert result.ok is True

    def test_verify_survives_close_and_reopen(
        self, db_path, fernet_key, valid_write_kwargs, analyst_ctx
    ):
        with AuditStore(db_path, fernet_key) as s:
            for i in range(5):
                s.write_record(**{**valid_write_kwargs, "content": f"msg {i}"})
        with AuditStore(db_path, fernet_key) as s2:
            result = s2.verify_chain(analyst_id=analyst_ctx["analyst_id"])
            assert result.ok is True

    def test_conversation_id_filter_returns_only_matching_records(
        self, store, valid_write_kwargs, analyst_ctx
    ):
        # ids 1,3,5 in conv-a; ids 2,4 in conv-b.
        for i in range(5):
            conv = "conv-a" if i % 2 == 0 else "conv-b"
            store.write_record(
                **{
                    **valid_write_kwargs,
                    "conversation_id": conv,
                    "content": f"msg {i}",
                }
            )
        records = store.get_records(**analyst_ctx, conversation_id="conv-a")
        assert [r.id for r in records] == [1, 3, 5]
        assert all(r.conversation_id == "conv-a" for r in records)


class TestAuditTamper:
    """Three mutation modes on audit_records are each caught and
    attributed to the exact row. The access_log chain stays intact."""

    @staticmethod
    def _write_n(store, valid_write_kwargs, n):
        for i in range(n):
            store.write_record(**{**valid_write_kwargs, "content": f"msg {i}"})

    def test_content_tamper_attributed_to_row(
        self, store, valid_write_kwargs, analyst_ctx, tamper
    ):
        self._write_n(store, valid_write_kwargs, 5)
        tamper("audit_records", 3, "content", "tampered_ciphertext")
        result = store.verify_chain(analyst_id=analyst_ctx["analyst_id"])
        assert result.ok is False
        assert result.audit_tampered_at == 3
        assert result.access_log_tampered_at is None

    def test_prev_hash_tamper_attributed_to_row(
        self, store, valid_write_kwargs, analyst_ctx, tamper
    ):
        self._write_n(store, valid_write_kwargs, 5)
        tamper("audit_records", 3, "prev_hash", "0" * 64)
        result = store.verify_chain(analyst_id=analyst_ctx["analyst_id"])
        assert result.ok is False
        assert result.audit_tampered_at == 3
        assert result.access_log_tampered_at is None

    def test_row_hash_tamper_attributed_to_row(
        self, store, valid_write_kwargs, analyst_ctx, tamper
    ):
        self._write_n(store, valid_write_kwargs, 5)
        tamper("audit_records", 3, "row_hash", "f" * 64)
        result = store.verify_chain(analyst_id=analyst_ctx["analyst_id"])
        assert result.ok is False
        assert result.audit_tampered_at == 3
        assert result.access_log_tampered_at is None


class TestAccessLogTamper:
    """Mutation on access_log is attributed to the access_log chain.
    The audit chain reports clean. The two chains are independent."""

    def test_access_log_mutation_does_not_taint_audit_chain(
        self, store, valid_write_kwargs, analyst_ctx, tamper
    ):
        for i in range(3):
            store.write_record(**{**valid_write_kwargs, "content": f"msg {i}"})
        store.get_records(**analyst_ctx)
        store.get_records(**analyst_ctx)
        # access_log now has two rows from the two get_records calls.
        tamper("access_log", 1, "justification", "tampered justification")
        result = store.verify_chain(analyst_id=analyst_ctx["analyst_id"])
        assert result.ok is False
        assert result.access_log_tampered_at == 1
        assert result.audit_tampered_at is None


class TestAccessLogOnReadPaths:
    """Every read path produces an access_log entry. The captured
    fields land in the correct columns and the entry shape is precise."""

    def test_successful_get_records_writes_correct_access_log(
        self, store, valid_write_kwargs, analyst_ctx, raw_read
    ):
        store.write_record(**valid_write_kwargs)
        store.write_record(**{**valid_write_kwargs, "content": "msg 2"})
        records = store.get_records(**analyst_ctx)
        rows = raw_read(
            "SELECT analyst_id, justification, row_count_returned, "
            "record_ids_accessed, success, error_message FROM access_log"
        )
        assert len(rows) == 1
        analyst_id, justification, count, ids_json, success, error_msg = rows[0]
        assert analyst_id == analyst_ctx["analyst_id"]
        assert justification == analyst_ctx["justification"]
        assert count == 2
        assert json.loads(ids_json) == [r.id for r in records]
        assert success == 1
        assert error_msg is None

    def test_verify_chain_writes_access_log_entry(
        self, store, valid_write_kwargs, analyst_ctx, raw_read
    ):
        store.write_record(**valid_write_kwargs)
        store.verify_chain(analyst_id=analyst_ctx["analyst_id"])
        rows = raw_read(
            "SELECT row_count_returned, record_ids_accessed, success FROM access_log"
        )
        assert len(rows) == 1
        count, ids_json, success = rows[0]
        assert count == 1
        assert json.loads(ids_json) == [1]
        assert success == 1

    def test_failed_get_records_writes_sanitized_error(
        self, store, valid_write_kwargs, analyst_ctx, raw_read, tamper
    ):
        store.write_record(**valid_write_kwargs)
        # A Fernet token from a different key forces InvalidToken on decrypt.
        foreign_token = Fernet(Fernet.generate_key()).encrypt(b"x").decode("ascii")
        tamper("audit_records", 1, "content", foreign_token)
        with pytest.raises(DecryptionError):
            store.get_records(**analyst_ctx)
        rows = raw_read("SELECT success, error_message FROM access_log")
        assert len(rows) == 1
        success, error_message = rows[0]
        assert success == 0
        # Only the exception type name is allowed in the access log.
        assert error_message == "DecryptionError"

    def test_query_params_json_scope(self, store, valid_write_kwargs, raw_read):
        store.write_record(**valid_write_kwargs)
        store.get_records(
            last=10,
            conversation_id="conv-1",
            since=datetime(2025, 1, 1, tzinfo=timezone.utc),
            until=datetime(2026, 12, 31, tzinfo=timezone.utc),
            analyst_id="analyst-2",
            justification="scope test",
            case_ref="CASE-42",
            source_ip="1.2.3.4",
            session_id="s-1",
        )
        rows = raw_read(
            "SELECT query_params_json, case_ref, source_ip, session_id FROM access_log"
        )
        assert len(rows) == 1
        query_params_json, case_ref, source_ip, session_id = rows[0]
        params = json.loads(query_params_json)
        # query_params_json carries only the four query filter keys.
        assert set(params.keys()) == {"last", "conversation_id", "since", "until"}
        # Context fields live in their own columns.
        assert case_ref == "CASE-42"
        assert source_ip == "1.2.3.4"
        assert session_id == "s-1"


class TestEncryption:
    """Plaintext round-trips correctly and never appears in the raw
    bytes on disk."""

    MARKER = "FORENSIC_TEST_SECRET_42"

    def test_round_trip_preserves_content_and_metadata(
        self, store, valid_write_kwargs, analyst_ctx
    ):
        kwargs = {
            **valid_write_kwargs,
            "content": self.MARKER,
            "metadata": {"secret_field": self.MARKER, "list": [1, 2, 3]},
        }
        store.write_record(**kwargs)
        records = store.get_records(**analyst_ctx)
        assert records[0].content == self.MARKER
        assert records[0].metadata == {"secret_field": self.MARKER, "list": [1, 2, 3]}

    def test_plaintext_absent_from_raw_columns(
        self, store, valid_write_kwargs, raw_read
    ):
        kwargs = {
            **valid_write_kwargs,
            "content": self.MARKER,
            "metadata": {"secret_field": self.MARKER},
        }
        store.write_record(**kwargs)
        rows = raw_read("SELECT content, metadata_json FROM audit_records")
        content_raw, metadata_raw = rows[0]
        # The columns are declared TEXT, so str is expected, but guard
        # against any sqlite3 configuration that returns bytes.
        for raw in (content_raw, metadata_raw):
            if isinstance(raw, bytes):
                assert self.MARKER.encode("utf-8") not in raw
            else:
                assert self.MARKER not in raw

    def test_rich_metadata_round_trip(
        self, store, valid_write_kwargs, analyst_ctx
    ):
        metadata = {
            "detector_hits": [
                {"detector": "presidio", "type": "PERSON", "score": 0.85, "start": 12, "end": 19},
                {"detector": "presidio", "type": "EMAIL", "score": 0.95, "start": 30, "end": 45},
            ],
            "model_response_time_ms": 1247,
            "user_locale": "en-US",
            "client_version": "claude-cli/2.1.126",
            "unicode_field": "héllo wörld 你好",
            "nested": {"a": [1, 2, 3], "b": {"c": None}},
        }
        store.write_record(**{**valid_write_kwargs, "metadata": metadata})
        records = store.get_records(**analyst_ctx)
        assert records[0].metadata == metadata

    def test_empty_metadata_round_trip(
        self, store, valid_write_kwargs, analyst_ctx
    ):
        store.write_record(**{**valid_write_kwargs, "metadata": {}})
        records = store.get_records(**analyst_ctx)
        assert records[0].metadata == {}


class TestWriteRecordValidation:
    """Invalid input is rejected with ValueError before any state
    change."""

    @pytest.mark.parametrize(
        ("field", "bad_value"),
        [
            ("source", "webhook"),
            ("role", "narrator"),
            ("detection_status", "yellow"),
            ("conversation_id", ""),
            ("model", ""),
        ],
    )
    def test_invalid_input_rejected_no_side_effect(
        self, store, valid_write_kwargs, raw_read, field, bad_value
    ):
        before = raw_read("SELECT COUNT(*) FROM audit_records")[0][0]
        with pytest.raises(ValueError):
            store.write_record(**{**valid_write_kwargs, field: bad_value})
        after = raw_read("SELECT COUNT(*) FROM audit_records")[0][0]
        assert after == before

    def test_id_sequence_after_rejected_write(
        self, store, valid_write_kwargs, analyst_ctx
    ):
        first_id = store.write_record(**valid_write_kwargs)
        with pytest.raises(ValueError):
            store.write_record(**{**valid_write_kwargs, "source": "webhook"})
        next_id = store.write_record(
            **{**valid_write_kwargs, "content": "after rejected"}
        )
        assert first_id == 1
        assert next_id == 2
        result = store.verify_chain(analyst_id=analyst_ctx["analyst_id"])
        assert result.ok is True


class TestGetRecordsValidation:
    """Validation on the read path raises ValueError before logging.
    Rejected calls produce no access_log entry."""

    @pytest.mark.parametrize(
        "overrides",
        [
            {"analyst_id": ""},
            {"justification": ""},
            {"last": -1},
            {"since": datetime(2025, 1, 1)},
            {"until": datetime(2025, 1, 1)},
        ],
    )
    def test_invalid_input_rejected_with_no_access_log(
        self, store, valid_write_kwargs, analyst_ctx, raw_read, overrides
    ):
        store.write_record(**valid_write_kwargs)
        kwargs = {**analyst_ctx, **overrides}
        with pytest.raises(ValueError):
            store.get_records(**kwargs)
        rows = raw_read("SELECT COUNT(*) FROM access_log")
        assert rows[0][0] == 0

    def test_aware_datetime_accepted(self, store, valid_write_kwargs, analyst_ctx):
        store.write_record(**valid_write_kwargs)
        records = store.get_records(
            **analyst_ctx,
            since=datetime(2020, 1, 1, tzinfo=timezone.utc),
            until=datetime(2030, 1, 1, tzinfo=timezone.utc),
        )
        assert len(records) == 1


class TestDecryptionError:
    """Corrupted ciphertext is caught loudly. The exception type name,
    and only the exception type name, makes it to the access log. The
    chain layer also catches the corruption (defense in depth)."""

    def test_decryption_error_on_foreign_token(
        self, store, valid_write_kwargs, analyst_ctx, tamper
    ):
        store.write_record(**valid_write_kwargs)
        foreign_token = Fernet(Fernet.generate_key()).encrypt(b"x").decode("ascii")
        tamper("audit_records", 1, "content", foreign_token)
        with pytest.raises(DecryptionError):
            store.get_records(**analyst_ctx)

    def test_access_log_entry_is_sanitized(
        self, store, valid_write_kwargs, analyst_ctx, raw_read, tamper
    ):
        store.write_record(**valid_write_kwargs)
        foreign_token = Fernet(Fernet.generate_key()).encrypt(b"x").decode("ascii")
        tamper("audit_records", 1, "content", foreign_token)
        with pytest.raises(DecryptionError):
            store.get_records(**analyst_ctx)
        rows = raw_read("SELECT error_message FROM access_log")
        assert rows[0][0] == "DecryptionError"

    def test_chain_breaks_after_content_tamper(
        self, store, valid_write_kwargs, analyst_ctx, tamper
    ):
        # The chain commits to the ciphertext, so substituting a foreign
        # token also breaks row_hash recomputation. Both the decryption
        # layer and the chain layer catch the corruption.
        store.write_record(**valid_write_kwargs)
        foreign_token = Fernet(Fernet.generate_key()).encrypt(b"x").decode("ascii")
        tamper("audit_records", 1, "content", foreign_token)
        result = store.verify_chain(analyst_id=analyst_ctx["analyst_id"])
        assert result.ok is False
        assert result.audit_tampered_at == 1


class TestCiphertextProperty:
    """Chain integrity is independent of the encryption key. An
    investigator without the key can still verify the audit trail."""

    def test_verify_chain_succeeds_with_different_key(
        self, db_path, fernet_key, valid_write_kwargs, analyst_ctx
    ):
        with AuditStore(db_path, fernet_key) as writer:
            for i in range(3):
                writer.write_record(**{**valid_write_kwargs, "content": f"msg {i}"})
        wrong_key = Fernet.generate_key()
        with AuditStore(db_path, wrong_key) as verifier:
            result = verifier.verify_chain(analyst_id=analyst_ctx["analyst_id"])
            assert result.ok is True

    def test_wrong_key_cannot_decrypt(
        self, db_path, fernet_key, valid_write_kwargs, analyst_ctx
    ):
        with AuditStore(db_path, fernet_key) as writer:
            writer.write_record(**valid_write_kwargs)
        wrong_key = Fernet.generate_key()
        with AuditStore(db_path, wrong_key) as reader:
            with pytest.raises(DecryptionError):
                reader.get_records(**analyst_ctx)


class TestAtomicity:
    """A transaction failure mid-write leaves no partial state, does
    not consume an id, and the next valid write resumes the chain."""

    def test_failed_insert_rolls_back_and_next_id_is_unchanged(
        self,
        store,
        valid_write_kwargs,
        analyst_ctx,
        raw_read,
        monkeypatch,
    ):
        first_id = store.write_record(**valid_write_kwargs)
        assert first_id == 1
        count_before = raw_read("SELECT COUNT(*) FROM audit_records")[0][0]
        assert count_before == 1

        # Wrap the connection so the next INSERT INTO audit_records raises.
        real_conn = store._conn
        monkeypatch.setattr(store, "_conn", _FailingInsertConn(real_conn))

        with pytest.raises(sqlite3.OperationalError):
            store.write_record(**{**valid_write_kwargs, "content": "should fail"})

        count_after_fail = raw_read("SELECT COUNT(*) FROM audit_records")[0][0]
        assert count_after_fail == count_before

        # Restore the real connection so the next write completes normally.
        monkeypatch.undo()

        next_id = store.write_record(
            **{**valid_write_kwargs, "content": "after recovery"}
        )
        # The failed write did not consume id=2.
        assert next_id == 2

        # Both stored rows still form a valid chain.
        result = store.verify_chain(analyst_id=analyst_ctx["analyst_id"])
        assert result.ok is True


class TestKeyVersion:
    """The configured key_version is persisted and surfaces on read."""

    def test_key_version_round_trips(
        self, db_path, fernet_key, valid_write_kwargs, analyst_ctx
    ):
        with AuditStore(db_path, fernet_key, key_version=2) as store:
            store.write_record(**valid_write_kwargs)
            records = store.get_records(**analyst_ctx)
            assert len(records) == 1
            assert records[0].key_version == 2
