"""Tamper-evident audit store for LLM prompt and response records.

This module is the forensic core of llmdr. It defines:

* :class:`AuditStore`, a SQLite-backed store with two independent hash chains
  (one for audit records, one for the access log).
* :class:`AuditRecord`, the decrypted view of a single audit row.
* :class:`VerifyResult`, the outcome of a chain integrity check.

Design notes (see CLAUDE.md for the full rationale):

* Every read of audit records writes an access log entry. There is no
  bypass mode. The access log itself is hash chained, separately from
  audit records, so it cannot be silently rewritten.
* Content and metadata are encrypted at rest with Fernet. The hash chain
  commits to the encrypted ciphertext, so chain verification does not
  require the encryption key.
* All database operations are plain sqlite3. No ORM, by deliberate
  forensic-transparency choice.

Filter scoping for access_log:
    The ``query_params_json`` column captures only the filters that
    select audit records: ``last``, ``conversation_id``, ``since``,
    ``until``. The remaining access context (``case_ref``, ``source_ip``,
    ``session_id``) lives in its own dedicated column. This split keeps
    the query_params_json field a clean, queryable record of "what
    audit data was selected" without conflating it with "who selected
    it and from where".
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from cryptography.fernet import Fernet, InvalidToken

logger = logging.getLogger(__name__)

GENESIS_HASH = "0" * 64
"""Sentinel prev_hash for the first record in either chain."""

Source = Literal["live", "import"]
Role = Literal["user", "assistant"]
DetectionStatus = Literal["ran_clean", "ran_with_hits", "skipped", "failed"]

_VALID_SOURCES: tuple[str, ...] = ("live", "import")
_VALID_ROLES: tuple[str, ...] = ("user", "assistant")
_VALID_DETECTION_STATUSES: tuple[str, ...] = (
    "ran_clean",
    "ran_with_hits",
    "skipped",
    "failed",
)
_KNOWN_TABLES: frozenset[str] = frozenset({"audit_records", "access_log"})


_SCHEMA_AUDIT_RECORDS = """
CREATE TABLE IF NOT EXISTS audit_records (
    id INTEGER PRIMARY KEY,
    ts TEXT NOT NULL,
    source TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    request_id TEXT,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    model TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    detection_status TEXT NOT NULL,
    key_version INTEGER NOT NULL DEFAULT 1,
    prev_hash TEXT NOT NULL,
    row_hash TEXT NOT NULL
)
"""

_SCHEMA_ACCESS_LOG = """
CREATE TABLE IF NOT EXISTS access_log (
    id INTEGER PRIMARY KEY,
    ts TEXT NOT NULL,
    analyst_id TEXT NOT NULL,
    query_params_json TEXT NOT NULL,
    justification TEXT NOT NULL,
    case_ref TEXT,
    row_count_returned INTEGER NOT NULL,
    record_ids_accessed TEXT NOT NULL,
    success INTEGER NOT NULL,
    error_message TEXT,
    source_ip TEXT,
    session_id TEXT,
    prev_hash TEXT NOT NULL,
    row_hash TEXT NOT NULL
)
"""

_INDEXES: tuple[str, ...] = (
    "CREATE INDEX IF NOT EXISTS idx_audit_records_conversation_id "
    "ON audit_records(conversation_id)",
    "CREATE INDEX IF NOT EXISTS idx_audit_records_ts ON audit_records(ts)",
    "CREATE INDEX IF NOT EXISTS idx_access_log_analyst_id ON access_log(analyst_id)",
    "CREATE INDEX IF NOT EXISTS idx_access_log_ts ON access_log(ts)",
)


# Columns hashed for audit_records, in canonical order. Excludes row_hash.
_AUDIT_HASH_COLUMNS: tuple[str, ...] = (
    "id",
    "ts",
    "source",
    "conversation_id",
    "request_id",
    "role",
    "content",
    "model",
    "metadata_json",
    "detection_status",
    "key_version",
    "prev_hash",
)

# Columns hashed for access_log, in canonical order. Excludes row_hash.
_ACCESS_HASH_COLUMNS: tuple[str, ...] = (
    "id",
    "ts",
    "analyst_id",
    "query_params_json",
    "justification",
    "case_ref",
    "row_count_returned",
    "record_ids_accessed",
    "success",
    "error_message",
    "source_ip",
    "session_id",
    "prev_hash",
)


@dataclass(frozen=True)
class AuditRecord:
    """Decrypted view of one row in audit_records."""

    id: int
    ts: str
    source: str
    conversation_id: str
    request_id: str | None
    role: str
    content: str
    model: str
    metadata: dict[str, Any]
    detection_status: str
    key_version: int
    prev_hash: str
    row_hash: str


@dataclass(frozen=True)
class VerifyResult:
    """Outcome of a chain integrity verification.

    ``ok`` is True only when both chains are intact. The two
    ``*_tampered_at`` fields hold the id of the first row whose stored
    row_hash or prev_hash does not match the recomputed value, or None
    when that chain verifies cleanly.
    """

    ok: bool
    audit_tampered_at: int | None
    access_log_tampered_at: int | None
    message: str


class AuditStoreError(Exception):
    """Base exception for AuditStore failures."""


class DecryptionError(AuditStoreError):
    """Raised when an audit record cannot be decrypted."""


def _canonical_json(obj: dict[str, Any]) -> bytes:
    """Encode ``obj`` as canonical JSON bytes for hashing.

    Canonical form: keys sorted, no insignificant whitespace, UTF-8 bytes.
    The same shape is used on write and on verification.
    """
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _format_dt(dt: datetime) -> str:
    """Format a timezone-aware datetime for ts comparison.

    Naive datetimes are rejected at the public boundary (:meth:`AuditStore.get_records`),
    so this helper assumes tzinfo is set and asserts the contract.
    """
    assert dt.tzinfo is not None, "internal: naive datetime reached _format_dt"
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class AuditStore:
    """SQLite-backed, hash-chained, encrypted audit store.

    Maintains two independent hash chains:

    * ``audit_records`` for prompt/response pairs. Content and metadata
      are encrypted with Fernet. The chain commits to the ciphertext, so
      verification does not require the key.
    * ``access_log`` for every read against the audit store. Plaintext
      because no sensitive content lives there. Mandatory for every
      :meth:`get_records` and :meth:`verify_chain` call.

    Use as a context manager to ensure the underlying connection is
    closed::

        with AuditStore(db_path, key) as store:
            store.write_record(...)

    Thread safety:
        AuditStore is **not** thread-safe. It wraps a single sqlite3
        connection and performs multi-statement transactions
        (``BEGIN IMMEDIATE`` ... ``COMMIT``) that must not interleave.
        Callers using FastAPI, asyncio, or any other concurrent
        framework must either instantiate a fresh AuditStore per request
        or guard a shared instance with an external lock. v0.1 does not
        provide a built-in pool or locking layer; this is a deliberate
        scope decision.
    """

    def __init__(
        self,
        db_path: str | Path,
        encryption_key: bytes,
        key_version: int = 1,
    ) -> None:
        """Open or create the audit store at ``db_path``.

        Args:
            db_path: Path to the SQLite database file. Created if missing.
            encryption_key: A url-safe base64 Fernet key. The caller is
                responsible for loading this from a secure source (env
                var). It is never persisted by this class.
            key_version: Logical key version label written on every new
                record. Reserved for future key rotation. Defaults to 1.
        """
        self._db_path = Path(db_path)
        self._fernet = Fernet(encryption_key)
        self._key_version = key_version
        # isolation_level=None gives explicit transaction control via BEGIN/COMMIT.
        self._conn = sqlite3.connect(str(self._db_path), isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        # WAL improves read concurrency and gives us reasonable durability.
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA synchronous = NORMAL")
        self._initialize_schema()
        logger.info("Opened audit store at %s", self._db_path)

    def __enter__(self) -> AuditStore:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()

    def _initialize_schema(self) -> None:
        cur = self._conn.cursor()
        try:
            cur.execute("BEGIN")
            cur.execute(_SCHEMA_AUDIT_RECORDS)
            cur.execute(_SCHEMA_ACCESS_LOG)
            for stmt in _INDEXES:
                cur.execute(stmt)
            cur.execute("COMMIT")
        except sqlite3.Error:
            cur.execute("ROLLBACK")
            raise

    @staticmethod
    def _last_hash(cur: sqlite3.Cursor, table: str) -> str:
        # Inline table name is safe: caller must pass a known table name.
        assert table in _KNOWN_TABLES, f"unknown table: {table!r}"
        row = cur.execute(
            f"SELECT row_hash FROM {table} ORDER BY id DESC LIMIT 1"  # noqa: S608
        ).fetchone()
        return row["row_hash"] if row is not None else GENESIS_HASH

    @staticmethod
    def _next_id(cur: sqlite3.Cursor, table: str) -> int:
        assert table in _KNOWN_TABLES, f"unknown table: {table!r}"
        row = cur.execute(
            f"SELECT COALESCE(MAX(id), 0) AS max_id FROM {table}"  # noqa: S608
        ).fetchone()
        return int(row["max_id"]) + 1

    def write_record(
        self,
        *,
        source: Source,
        conversation_id: str,
        request_id: str | None,
        role: Role,
        content: str,
        model: str,
        metadata: dict[str, Any],
        detection_status: DetectionStatus,
    ) -> int:
        """Append a single audit record to the chain.

        Encrypts ``content`` and ``metadata`` with Fernet, computes
        ``prev_hash`` from the current chain head, computes ``row_hash``
        over the canonical JSON of all non-hash columns, and inserts the
        row inside a single ``BEGIN IMMEDIATE`` transaction. Either the
        full record is written or nothing is written.

        Args:
            source: Origin of the record. ``"live"`` for proxied traffic,
                ``"import"`` for ingested conversation exports.
            conversation_id: Stable identifier grouping prompts and
                responses from one logical conversation. Required, must
                be non-empty.
            request_id: Anthropic API request id from response headers,
                or None when not available (e.g. import mode).
            role: ``"user"`` or ``"assistant"``.
            content: Plaintext message content. Will be encrypted before
                it touches disk.
            model: Model identifier reported by the provider. Required,
                must be non-empty.
            metadata: Arbitrary JSON-serializable dict, encrypted at
                rest. Detector hits live here.
            detection_status: One of ``ran_clean``, ``ran_with_hits``,
                ``skipped``, ``failed``. Distinguishes "no hits" from
                "detection did not run".

        Returns:
            The id of the inserted record.

        Raises:
            ValueError: If an enum-like argument is invalid, or if
                ``conversation_id`` or ``model`` is empty.
            sqlite3.Error: If the transaction fails. The transaction is
                rolled back before the exception propagates.
        """
        if source not in _VALID_SOURCES:
            raise ValueError(f"invalid source: {source!r}")
        if role not in _VALID_ROLES:
            raise ValueError(f"invalid role: {role!r}")
        if detection_status not in _VALID_DETECTION_STATUSES:
            raise ValueError(f"invalid detection_status: {detection_status!r}")
        if not conversation_id:
            raise ValueError("conversation_id is required")
        if not model:
            raise ValueError("model is required")

        ts = _utc_now_iso()
        encrypted_content = self._fernet.encrypt(content.encode("utf-8")).decode("ascii")
        metadata_bytes = json.dumps(metadata, sort_keys=True, ensure_ascii=False).encode("utf-8")
        encrypted_metadata = self._fernet.encrypt(metadata_bytes).decode("ascii")

        cur = self._conn.cursor()
        try:
            cur.execute("BEGIN IMMEDIATE")
            prev_hash = self._last_hash(cur, "audit_records")
            next_id = self._next_id(cur, "audit_records")

            row_for_hash: dict[str, Any] = {
                "id": next_id,
                "ts": ts,
                "source": source,
                "conversation_id": conversation_id,
                "request_id": request_id,
                "role": role,
                "content": encrypted_content,
                "model": model,
                "metadata_json": encrypted_metadata,
                "detection_status": detection_status,
                "key_version": self._key_version,
                "prev_hash": prev_hash,
            }
            # Defensive check: the hashed dict must contain exactly the
            # expected columns. A drift here would silently break the chain.
            if set(row_for_hash.keys()) != set(_AUDIT_HASH_COLUMNS):
                raise AuditStoreError("internal error: audit hash column set drift")
            row_hash = _sha256_hex(_canonical_json(row_for_hash))

            cur.execute(
                """
                INSERT INTO audit_records (
                    id, ts, source, conversation_id, request_id, role, content,
                    model, metadata_json, detection_status, key_version,
                    prev_hash, row_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    next_id,
                    ts,
                    source,
                    conversation_id,
                    request_id,
                    role,
                    encrypted_content,
                    model,
                    encrypted_metadata,
                    detection_status,
                    self._key_version,
                    prev_hash,
                    row_hash,
                ),
            )
            cur.execute("COMMIT")
        except Exception:
            cur.execute("ROLLBACK")
            raise

        logger.info(
            "wrote audit_records id=%d source=%s role=%s detection=%s",
            next_id,
            source,
            role,
            detection_status,
        )
        return next_id

    def get_records(
        self,
        *,
        last: int | None = None,
        conversation_id: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        analyst_id: str,
        justification: str,
        case_ref: str | None = None,
        source_ip: str | None = None,
        session_id: str | None = None,
    ) -> list[AuditRecord]:
        """Retrieve decrypted audit records and record the access.

        Every call writes one access_log entry, whether the underlying
        query succeeds or fails. The entry captures the analyst
        identity, the filters used, the count and ids of records
        returned, and on failure the exception type name only (never
        the exception message, to keep the access_log free of any
        accidental content leak).

        The ``query_params_json`` column on the access_log row records
        only ``last``, ``conversation_id``, ``since``, ``until``. The
        ``case_ref``, ``source_ip``, and ``session_id`` arguments are
        stored in their own access_log columns rather than mixed into
        ``query_params_json``.

        Args:
            last: Return at most this many of the most recent records.
                Must be non-negative when provided.
            conversation_id: Restrict to a single conversation.
            since: Inclusive lower bound on ``ts``. Must be
                timezone-aware. Naive datetimes are rejected.
            until: Inclusive upper bound on ``ts``. Must be
                timezone-aware. Naive datetimes are rejected.
            analyst_id: Required identity of the reader. Empty string
                rejected.
            justification: Required free-text business justification.
                Empty string rejected.
            case_ref: Optional case or ticket reference.
            source_ip: Optional source IP. Stored on the access_log row.
            session_id: Optional session id. Stored on the access_log row.

        Returns:
            Decrypted records in ascending id order.

        Raises:
            ValueError: If ``analyst_id`` or ``justification`` is empty,
                if ``last`` is negative, or if ``since`` or ``until`` is
                naive (lacks tzinfo).
            DecryptionError: If a row cannot be decrypted with the
                current key. An access_log entry recording the failure
                is written before the exception propagates.
            sqlite3.Error: On database failure. An access_log entry is
                written on best effort before the exception propagates.
        """
        if not analyst_id:
            raise ValueError("analyst_id is required")
        if not justification:
            raise ValueError("justification is required")
        if since is not None and since.tzinfo is None:
            raise ValueError(
                "since must be a timezone-aware datetime; naive datetimes are rejected"
            )
        if until is not None and until.tzinfo is None:
            raise ValueError(
                "until must be a timezone-aware datetime; naive datetimes are rejected"
            )
        if last is not None and last < 0:
            raise ValueError("last must be non-negative")

        query_params: dict[str, Any] = {
            "last": last,
            "conversation_id": conversation_id,
            "since": _format_dt(since) if since is not None else None,
            "until": _format_dt(until) if until is not None else None,
        }

        try:
            records = self._query_records(
                last=last,
                conversation_id=conversation_id,
                since=since,
                until=until,
            )
        except Exception as exc:
            # Sanitized: only the exception type name reaches access_log,
            # never str(exc), which could carry user content from sqlite or
            # decryption errors. Detail belongs in logger output, not in
            # the persisted plaintext audit trail.
            self._write_access_log(
                analyst_id=analyst_id,
                query_params=query_params,
                justification=justification,
                case_ref=case_ref,
                row_count=0,
                record_ids=[],
                success=False,
                error_message=type(exc).__name__,
                source_ip=source_ip,
                session_id=session_id,
            )
            raise

        self._write_access_log(
            analyst_id=analyst_id,
            query_params=query_params,
            justification=justification,
            case_ref=case_ref,
            row_count=len(records),
            record_ids=[r.id for r in records],
            success=True,
            error_message=None,
            source_ip=source_ip,
            session_id=session_id,
        )
        return records

    def _query_records(
        self,
        *,
        last: int | None,
        conversation_id: str | None,
        since: datetime | None,
        until: datetime | None,
    ) -> list[AuditRecord]:
        clauses: list[str] = []
        params: list[Any] = []
        if conversation_id is not None:
            clauses.append("conversation_id = ?")
            params.append(conversation_id)
        if since is not None:
            clauses.append("ts >= ?")
            params.append(_format_dt(since))
        if until is not None:
            clauses.append("ts <= ?")
            params.append(_format_dt(until))

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        if last is not None:
            sql = (
                f"SELECT * FROM ("
                f"  SELECT * FROM audit_records {where} ORDER BY id DESC LIMIT ?"
                f") ORDER BY id ASC"
            )
            params.append(last)
        else:
            sql = f"SELECT * FROM audit_records {where} ORDER BY id ASC"

        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_record(r) for r in rows]

    def _row_to_record(self, row: sqlite3.Row) -> AuditRecord:
        try:
            content = self._fernet.decrypt(row["content"].encode("ascii")).decode("utf-8")
            metadata_bytes = self._fernet.decrypt(row["metadata_json"].encode("ascii"))
            metadata = json.loads(metadata_bytes.decode("utf-8"))
        except InvalidToken as exc:
            raise DecryptionError(
                f"failed to decrypt audit_records.id={row['id']}: invalid Fernet token"
            ) from exc
        except json.JSONDecodeError as exc:
            raise DecryptionError(
                f"failed to parse decrypted metadata for audit_records.id={row['id']}"
            ) from exc

        return AuditRecord(
            id=row["id"],
            ts=row["ts"],
            source=row["source"],
            conversation_id=row["conversation_id"],
            request_id=row["request_id"],
            role=row["role"],
            content=content,
            model=row["model"],
            metadata=metadata,
            detection_status=row["detection_status"],
            key_version=row["key_version"],
            prev_hash=row["prev_hash"],
            row_hash=row["row_hash"],
        )

    def _write_access_log(
        self,
        *,
        analyst_id: str,
        query_params: dict[str, Any],
        justification: str,
        case_ref: str | None,
        row_count: int,
        record_ids: list[int],
        success: bool,
        error_message: str | None,
        source_ip: str | None,
        session_id: str | None,
    ) -> int:
        ts = _utc_now_iso()
        query_params_json = json.dumps(query_params, sort_keys=True, ensure_ascii=False)
        record_ids_json = json.dumps(record_ids)
        success_int = 1 if success else 0

        cur = self._conn.cursor()
        try:
            cur.execute("BEGIN IMMEDIATE")
            prev_hash = self._last_hash(cur, "access_log")
            next_id = self._next_id(cur, "access_log")

            row_for_hash: dict[str, Any] = {
                "id": next_id,
                "ts": ts,
                "analyst_id": analyst_id,
                "query_params_json": query_params_json,
                "justification": justification,
                "case_ref": case_ref,
                "row_count_returned": row_count,
                "record_ids_accessed": record_ids_json,
                "success": success_int,
                "error_message": error_message,
                "source_ip": source_ip,
                "session_id": session_id,
                "prev_hash": prev_hash,
            }
            if set(row_for_hash.keys()) != set(_ACCESS_HASH_COLUMNS):
                raise AuditStoreError("internal error: access_log hash column set drift")
            row_hash = _sha256_hex(_canonical_json(row_for_hash))

            cur.execute(
                """
                INSERT INTO access_log (
                    id, ts, analyst_id, query_params_json, justification,
                    case_ref, row_count_returned, record_ids_accessed, success,
                    error_message, source_ip, session_id, prev_hash, row_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    next_id,
                    ts,
                    analyst_id,
                    query_params_json,
                    justification,
                    case_ref,
                    row_count,
                    record_ids_json,
                    success_int,
                    error_message,
                    source_ip,
                    session_id,
                    prev_hash,
                    row_hash,
                ),
            )
            cur.execute("COMMIT")
        except Exception:
            cur.execute("ROLLBACK")
            raise

        logger.info(
            "wrote access_log id=%d analyst=%s success=%d row_count=%d",
            next_id,
            analyst_id,
            success_int,
            row_count,
        )
        return next_id

    def verify_chain(
        self,
        *,
        analyst_id: str,
        justification: str = "chain integrity verification",
    ) -> VerifyResult:
        """Verify both the audit_records and access_log hash chains.

        Walks each table in id order, recomputing every row_hash from
        stored fields and confirming each prev_hash matches the prior
        row's row_hash. Returns the id of the first divergence in each
        chain, or None when that chain is intact.

        Always writes one access_log entry recording this verification,
        with ``record_ids_accessed`` set to every audit record examined.
        Verification reads the encrypted ciphertext only; the encryption
        key is not exercised here.

        Args:
            analyst_id: Required identity of the caller.
            justification: Free-text justification. Defaults to
                ``"chain integrity verification"``.

        Returns:
            A :class:`VerifyResult`. ``ok`` is True only when both
            chains verify cleanly.

        Raises:
            ValueError: If ``analyst_id`` is empty.
        """
        if not analyst_id:
            raise ValueError("analyst_id is required")

        audit_tampered_at = self._verify_audit_records()
        access_tampered_at = self._verify_access_log()

        ok = audit_tampered_at is None and access_tampered_at is None
        if ok:
            message = "chain intact"
        else:
            parts: list[str] = []
            if audit_tampered_at is not None:
                parts.append(f"audit_records broken at id={audit_tampered_at}")
            if access_tampered_at is not None:
                parts.append(f"access_log broken at id={access_tampered_at}")
            message = "; ".join(parts)
            logger.error("chain verification failed: %s", message)

        all_ids = [
            int(r["id"])
            for r in self._conn.execute(
                "SELECT id FROM audit_records ORDER BY id ASC"
            ).fetchall()
        ]
        self._write_access_log(
            analyst_id=analyst_id,
            query_params={"operation": "verify_chain"},
            justification=justification,
            case_ref=None,
            row_count=len(all_ids),
            record_ids=all_ids,
            success=ok,
            error_message=None if ok else message,
            source_ip=None,
            session_id=None,
        )

        return VerifyResult(
            ok=ok,
            audit_tampered_at=audit_tampered_at,
            access_log_tampered_at=access_tampered_at,
            message=message,
        )

    def _verify_audit_records(self) -> int | None:
        prev = GENESIS_HASH
        cols = ", ".join(_AUDIT_HASH_COLUMNS) + ", row_hash"
        rows = self._conn.execute(
            f"SELECT {cols} FROM audit_records ORDER BY id ASC"  # noqa: S608
        ).fetchall()
        for row in rows:
            if row["prev_hash"] != prev:
                logger.error("audit_records prev_hash mismatch at id=%d", row["id"])
                return int(row["id"])
            row_for_hash = {col: row[col] for col in _AUDIT_HASH_COLUMNS}
            expected = _sha256_hex(_canonical_json(row_for_hash))
            if expected != row["row_hash"]:
                logger.error("audit_records row_hash mismatch at id=%d", row["id"])
                return int(row["id"])
            prev = row["row_hash"]
        return None

    def _verify_access_log(self) -> int | None:
        prev = GENESIS_HASH
        cols = ", ".join(_ACCESS_HASH_COLUMNS) + ", row_hash"
        rows = self._conn.execute(
            f"SELECT {cols} FROM access_log ORDER BY id ASC"  # noqa: S608
        ).fetchall()
        for row in rows:
            if row["prev_hash"] != prev:
                logger.error("access_log prev_hash mismatch at id=%d", row["id"])
                return int(row["id"])
            row_for_hash = {col: row[col] for col in _ACCESS_HASH_COLUMNS}
            expected = _sha256_hex(_canonical_json(row_for_hash))
            if expected != row["row_hash"]:
                logger.error("access_log row_hash mismatch at id=%d", row["id"])
                return int(row["id"])
            prev = row["row_hash"]
        return None
