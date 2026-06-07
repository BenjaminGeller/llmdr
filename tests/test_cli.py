"""Tests for the CLI orchestration layer.

The CLI is a thin orchestration of audit.py, config.py, and proxy.py,
each of which has its own suite (audit: 42, config: 63, proxy: 65,
totalling 170 tests). These tests pin CLI-specific behavior only:

* exit codes (0 ok, 1 tamper, 2 config/usage)
* stdout vs stderr separation
* flag handling and required-flag enforcement
* env-var resolution order (flag, then env, then default-or-error)
* side effects on the audit store
* the security-critical property that a malformed encryption key
  never appears in any user-visible CLI output

They do not re-test audit chain logic, encryption, key validation, or
proxy forwarding. Those have their own dedicated suites.

Class breakdown:

* TestKeygen
* TestServe (uvicorn.run is monkeypatched; no socket is ever opened)
* TestVerify
* TestShow
* TestImport
* TestCrossCutting (key-not-echoed, justification-never-from-env,
  consistent ConfigError handling across commands, help discoverability)
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

import pytest
from cryptography.fernet import Fernet
from typer.testing import CliRunner

from llmdr.audit import AuditStore
from llmdr.cli import app
from llmdr.config import (
    ENV_ANALYST_ID,
    ENV_DB_PATH,
    ENV_ENCRYPTION_KEY,
    ENV_LOG_LEVEL,
    ENV_PROXY_HOST,
    ENV_PROXY_PORT,
    ENV_UPSTREAM_BASE_URL,
)


# -----------------------------------------------------------------------------
# Fixtures and helpers
# -----------------------------------------------------------------------------


_ALL_ENV_VARS = (
    ENV_ENCRYPTION_KEY,
    ENV_DB_PATH,
    ENV_UPSTREAM_BASE_URL,
    ENV_PROXY_HOST,
    ENV_PROXY_PORT,
    ENV_LOG_LEVEL,
    ENV_ANALYST_ID,
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Clear every LLMDR_* env var the loader reads.

    Autouse so every test starts from a deterministic environment
    regardless of the host shell. Mirrors the same pattern used in
    test_config.py.
    """
    for var in _ALL_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def fernet_key() -> bytes:
    return Fernet.generate_key()


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "cli.db"


@pytest.fixture
def env_with_key_and_db(monkeypatch, fernet_key: bytes, db_path: Path) -> None:
    """Set the two env vars required by Config.from_env to succeed."""
    monkeypatch.setenv(ENV_ENCRYPTION_KEY, fernet_key.decode("ascii"))
    monkeypatch.setenv(ENV_DB_PATH, str(db_path))


@pytest.fixture
def runner() -> CliRunner:
    # click 8.3 separates stdout and stderr on the Result by default;
    # no mix_stderr argument is needed (or accepted).
    return CliRunner()


def _seed_pair(
    db_path: Path,
    key: bytes,
    *,
    conv_id: str = "conv-1",
    user_content: str = "user prompt content",
    assistant_content: str = "assistant reply content",
    model: str = "claude-opus-4-7",
) -> str:
    """Write one user + one assistant audit record. Returns conv_id."""
    with AuditStore(db_path, key) as store:
        store.write_record(
            source="live",
            conversation_id=conv_id,
            request_id=None,
            role="user",
            content=user_content,
            model=model,
            metadata={},
            detection_status="skipped",
        )
        store.write_record(
            source="live",
            conversation_id=conv_id,
            request_id="req-test",
            role="assistant",
            content=assistant_content,
            model=model,
            metadata={},
            detection_status="skipped",
        )
    return conv_id


def _seed_records(
    db_path: Path,
    key: bytes,
    pairs: list[tuple[str, str]],
) -> None:
    """Write a series of (conv_id, content) user records for filtering tests."""
    with AuditStore(db_path, key) as store:
        for conv_id, content in pairs:
            store.write_record(
                source="live",
                conversation_id=conv_id,
                request_id=None,
                role="user",
                content=content,
                model="claude-test",
                metadata={},
                detection_status="skipped",
            )


def _tamper(db_path: Path, table: str, row_id: int, column: str, value: Any) -> None:
    """Mutate one column on one row, bypassing AuditStore.

    Matches the test_audit.py tamper helper. Inlined here so test_cli.py
    is self-contained.
    """
    assert table in {"audit_records", "access_log"}
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            f"UPDATE {table} SET {column} = ? WHERE id = ?",  # noqa: S608
            (value, row_id),
        )
        conn.commit()
    finally:
        conn.close()


def _count_records(db_path: Path) -> tuple[int, int]:
    if not db_path.exists():
        return 0, 0
    conn = sqlite3.connect(str(db_path))
    try:
        a = conn.execute("SELECT COUNT(*) FROM audit_records").fetchone()[0]
        b = conn.execute("SELECT COUNT(*) FROM access_log").fetchone()[0]
        return int(a), int(b)
    finally:
        conn.close()


def _read_access_log(db_path: Path) -> list[sqlite3.Row]:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        return list(conn.execute("SELECT * FROM access_log ORDER BY id ASC"))
    finally:
        conn.close()


def _read_audit_records(db_path: Path, key: bytes):
    """Read records for assertions via AuditStore (uses a test analyst)."""
    with AuditStore(db_path, key) as store:
        return store.get_records(analyst_id="test-reader", justification="test")


def _make_export_file(
    tmp_path: Path,
    conversations: list[dict[str, Any]],
    *,
    filename: str = "export.json",
    top_level: str = "array",
) -> Path:
    """Write a Claude-style export JSON file and return its path."""
    payload: Any
    if top_level == "array":
        payload = conversations
    elif top_level == "single":
        assert len(conversations) == 1
        payload = conversations[0]
    else:
        raise ValueError(f"unknown top_level: {top_level!r}")
    path = tmp_path / filename
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_raw_file(tmp_path: Path, raw: str, *, filename: str = "export.json") -> Path:
    path = tmp_path / filename
    path.write_text(raw, encoding="utf-8")
    return path


def _make_conv(
    *,
    uuid: str = "conv-uuid-1",
    messages: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if messages is None:
        messages = [
            {
                "uuid": "msg-1",
                "text": "hello from human",
                "sender": "human",
                "created_at": "2026-06-01T10:00:00Z",
            },
            {
                "uuid": "msg-2",
                "text": "hello from assistant",
                "sender": "assistant",
                "created_at": "2026-06-01T10:00:01Z",
            },
        ]
    return {
        "uuid": uuid,
        "name": "Test Conversation",
        "created_at": "2026-06-01T10:00:00Z",
        "chat_messages": messages,
    }


# -----------------------------------------------------------------------------
# TestKeygen
# -----------------------------------------------------------------------------


class TestKeygen:
    """keygen has no Config dependency; tests run with the autouse clean env."""

    def test_stdout_is_a_valid_fernet_key(self, runner):
        result = runner.invoke(app, ["keygen"])
        assert result.exit_code == 0
        key_text = result.stdout.rstrip("\n")
        # Round-trip through Fernet constructor: invalid keys raise.
        Fernet(key_text.encode("ascii"))

    def test_stdout_is_exactly_key_plus_single_newline(self, runner):
        result = runner.invoke(app, ["keygen"])
        assert result.exit_code == 0
        # Standard Fernet key is 44 ASCII chars; output is 44 + 1 newline.
        assert result.stdout.endswith("\n")
        assert not result.stdout.startswith("\n")
        assert "\n\n" not in result.stdout
        key_text = result.stdout.rstrip("\n")
        assert len(key_text) == 44

    def test_stderr_contains_warning_by_default(self, runner):
        result = runner.invoke(app, ["keygen"])
        assert result.exit_code == 0
        assert "Handle this key like a password" in result.stderr
        # The shell-variable-leak rationale must be present; this is the
        # specific wording refinement that landed.
        assert "leak" in result.stderr.lower()
        assert "child processes" in result.stderr

    def test_quiet_suppresses_stderr_warning_but_stdout_unchanged(self, runner):
        result = runner.invoke(app, ["keygen", "--quiet"])
        assert result.exit_code == 0
        assert result.stderr == ""
        # stdout still carries a valid key.
        key_text = result.stdout.rstrip("\n")
        assert len(key_text) == 44
        Fernet(key_text.encode("ascii"))

    def test_works_with_no_env_set(self, runner):
        # clean_env autouse already removed everything. Confirm keygen
        # doesn't touch Config at all.
        assert os.environ.get(ENV_ENCRYPTION_KEY) is None
        result = runner.invoke(app, ["keygen", "--quiet"])
        assert result.exit_code == 0

    def test_two_invocations_produce_different_keys(self, runner):
        # Pins fresh randomness per invocation. A constant key would be
        # catastrophic for a forensic tool (every store encrypted
        # identically; one compromised key compromises every install).
        r1 = runner.invoke(app, ["keygen", "--quiet"])
        r2 = runner.invoke(app, ["keygen", "--quiet"])
        assert r1.exit_code == 0
        assert r2.exit_code == 0
        assert r1.stdout != r2.stdout


# -----------------------------------------------------------------------------
# TestServe
# -----------------------------------------------------------------------------


class TestServe:
    """serve handoff to uvicorn.run is monkeypatched. No socket is opened."""

    def test_serve_calls_uvicorn_with_config_host_port_log_level(
        self, runner, env_with_key_and_db, monkeypatch
    ):
        calls: list[tuple[tuple, dict]] = []

        def fake_run(*args, **kwargs):
            calls.append((args, kwargs))

        monkeypatch.setattr("llmdr.cli.uvicorn.run", fake_run)
        result = runner.invoke(app, ["serve"])
        assert result.exit_code == 0
        assert len(calls) == 1
        _, kwargs = calls[0]
        assert kwargs["host"] == "127.0.0.1"
        assert kwargs["port"] == 8080
        # log_level passed lower-cased per uvicorn's convention.
        assert kwargs["log_level"] == "info"
        # First positional arg is the FastAPI app from create_app.
        from fastapi import FastAPI
        assert isinstance(calls[0][0][0], FastAPI)

    def test_serve_with_loopback_host_does_not_emit_warning(
        self, runner, env_with_key_and_db, monkeypatch
    ):
        monkeypatch.setenv(ENV_PROXY_HOST, "127.0.0.1")
        monkeypatch.setattr("llmdr.cli.uvicorn.run", lambda *a, **kw: None)
        result = runner.invoke(app, ["serve"])
        assert result.exit_code == 0
        assert "WARNING:" not in result.stderr

    def test_serve_with_non_loopback_host_emits_warning_with_address(
        self, runner, env_with_key_and_db, monkeypatch
    ):
        monkeypatch.setenv(ENV_PROXY_HOST, "0.0.0.0")
        monkeypatch.setenv(ENV_PROXY_PORT, "9000")
        monkeypatch.setattr("llmdr.cli.uvicorn.run", lambda *a, **kw: None)
        result = runner.invoke(app, ["serve"])
        assert result.exit_code == 0
        assert "WARNING:" in result.stderr
        assert "0.0.0.0:9000" in result.stderr
        assert "non-loopback" in result.stderr

    def test_serve_without_encryption_key_exits_2_and_does_not_call_uvicorn(
        self, runner, monkeypatch
    ):
        calls: list = []

        def fake_run(*args, **kwargs):
            calls.append((args, kwargs))

        monkeypatch.setattr("llmdr.cli.uvicorn.run", fake_run)
        result = runner.invoke(app, ["serve"])
        assert result.exit_code == 2
        assert calls == []
        assert ENV_ENCRYPTION_KEY in result.stderr

    def test_serve_with_localhost_treated_as_loopback(
        self, runner, env_with_key_and_db, monkeypatch
    ):
        # Pins that all three loopback identifiers are honored.
        monkeypatch.setenv(ENV_PROXY_HOST, "localhost")
        monkeypatch.setattr("llmdr.cli.uvicorn.run", lambda *a, **kw: None)
        result = runner.invoke(app, ["serve"])
        assert result.exit_code == 0
        assert "WARNING:" not in result.stderr


# -----------------------------------------------------------------------------
# TestVerify
# -----------------------------------------------------------------------------


class TestVerify:
    """Exit codes, tamper detection, analyst_id resolution, access_log effect."""

    def test_intact_chain_exits_0_with_intact_message(
        self, runner, env_with_key_and_db, db_path, fernet_key
    ):
        _seed_pair(db_path, fernet_key)
        result = runner.invoke(
            app,
            ["verify", "-j", "scheduled audit", "--analyst-id", "alice"],
        )
        assert result.exit_code == 0
        assert "audit chain intact" in result.stdout
        assert "audit_records: 2 rows" in result.stdout

    def test_tampered_audit_records_exits_1_with_broken_message_and_id(
        self, runner, env_with_key_and_db, db_path, fernet_key
    ):
        _seed_pair(db_path, fernet_key)
        _tamper(db_path, "audit_records", 1, "content", "garbage-ciphertext")
        result = runner.invoke(
            app,
            ["verify", "-j", "investigation", "--analyst-id", "alice"],
        )
        assert result.exit_code == 1
        assert "audit chain BROKEN" in result.stdout
        assert "audit_records broken at id=1" in result.stdout

    def test_tampered_access_log_exits_1(
        self, runner, env_with_key_and_db, db_path, fernet_key
    ):
        _seed_pair(db_path, fernet_key)
        # Produce an access_log entry by running verify once.
        first = runner.invoke(
            app, ["verify", "-j", "first", "--analyst-id", "alice"]
        )
        assert first.exit_code == 0
        # Tamper it.
        _tamper(db_path, "access_log", 1, "analyst_id", "bob")
        result = runner.invoke(
            app, ["verify", "-j", "second", "--analyst-id", "alice"]
        )
        assert result.exit_code == 1
        assert "audit chain BROKEN" in result.stdout
        assert "access_log broken at id=1" in result.stdout

    def test_missing_justification_exits_2_usage_error(
        self, runner, env_with_key_and_db, db_path, fernet_key
    ):
        _seed_pair(db_path, fernet_key)
        result = runner.invoke(app, ["verify", "--analyst-id", "alice"])
        assert result.exit_code == 2
        # typer's usage error names the missing option.
        assert "justification" in result.stderr.lower()

    def test_missing_analyst_id_exits_2_with_explicit_message(
        self, runner, env_with_key_and_db, db_path, fernet_key
    ):
        _seed_pair(db_path, fernet_key)
        result = runner.invoke(app, ["verify", "-j", "test"])
        assert result.exit_code == 2
        assert "analyst-id required" in result.stderr

    def test_analyst_id_resolves_from_env(
        self, runner, env_with_key_and_db, db_path, fernet_key, monkeypatch
    ):
        _seed_pair(db_path, fernet_key)
        monkeypatch.setenv(ENV_ANALYST_ID, "carol")
        result = runner.invoke(app, ["verify", "-j", "env-test"])
        assert result.exit_code == 0
        rows = _read_access_log(db_path)
        # verify_chain wrote the most recent access_log entry; it must
        # carry the env-resolved analyst.
        assert rows[-1]["analyst_id"] == "carol"

    def test_verify_writes_one_access_log_entry(
        self, runner, env_with_key_and_db, db_path, fernet_key
    ):
        _seed_pair(db_path, fernet_key)
        before_audit, before_access = _count_records(db_path)
        result = runner.invoke(
            app, ["verify", "-j", "audit-effect", "--analyst-id", "alice"]
        )
        assert result.exit_code == 0
        after_audit, after_access = _count_records(db_path)
        assert after_audit == before_audit
        assert after_access == before_access + 1


# -----------------------------------------------------------------------------
# TestShow
# -----------------------------------------------------------------------------


_CONTENT_SENTINEL = "SECRET_CONTENT_SENTINEL_xyz789"


class TestShow:
    """Conservative defaults, content opt-in, JSON shape, filter behavior."""

    def test_empty_store_returns_no_records_message_exit_0(
        self, runner, env_with_key_and_db
    ):
        result = runner.invoke(
            app, ["show", "-j", "empty-check", "--analyst-id", "alice"]
        )
        assert result.exit_code == 0
        assert "(no records)" in result.stdout

    def test_default_output_omits_decrypted_content(
        self, runner, env_with_key_and_db, db_path, fernet_key
    ):
        _seed_pair(
            db_path, fernet_key,
            user_content=_CONTENT_SENTINEL,
            assistant_content="reply",
        )
        result = runner.invoke(
            app, ["show", "-j", "metadata-only", "--analyst-id", "alice"]
        )
        assert result.exit_code == 0
        # The conservative-default property: content never appears
        # without --full.
        assert _CONTENT_SENTINEL not in result.stdout
        # Metadata columns DO appear.
        assert "user" in result.stdout
        assert "assistant" in result.stdout

    def test_full_output_includes_decrypted_content(
        self, runner, env_with_key_and_db, db_path, fernet_key
    ):
        _seed_pair(
            db_path, fernet_key,
            user_content=_CONTENT_SENTINEL,
            assistant_content="reply",
        )
        result = runner.invoke(
            app,
            ["show", "-j", "content-needed", "--analyst-id", "alice", "--full"],
        )
        assert result.exit_code == 0
        assert _CONTENT_SENTINEL in result.stdout

    def test_full_emits_stderr_access_log_nudge_with_id(
        self, runner, env_with_key_and_db, db_path, fernet_key
    ):
        _seed_pair(db_path, fernet_key)
        result = runner.invoke(
            app,
            ["show", "-j", "content-needed", "--analyst-id", "alice", "--full"],
        )
        assert result.exit_code == 0
        assert "Decrypted content displayed" in result.stderr
        # The nudge pins the access_log id of this read; format is "id=N".
        assert "id=" in result.stderr

    def test_json_default_is_valid_json_with_hash_fields_no_content(
        self, runner, env_with_key_and_db, db_path, fernet_key
    ):
        _seed_pair(
            db_path, fernet_key,
            user_content=_CONTENT_SENTINEL,
            assistant_content="reply",
        )
        result = runner.invoke(
            app,
            ["show", "-j", "json-test", "--analyst-id", "alice", "--json"],
        )
        assert result.exit_code == 0
        parsed = json.loads(result.stdout)
        assert isinstance(parsed, list)
        assert len(parsed) == 2
        for item in parsed:
            # Intentional JSON asymmetry: hash fields exposed for
            # chain-verification scripting.
            assert "prev_hash" in item
            assert "row_hash" in item
            # Content NOT included without --full.
            assert "content" not in item

    def test_json_with_full_includes_content(
        self, runner, env_with_key_and_db, db_path, fernet_key
    ):
        _seed_pair(
            db_path, fernet_key,
            user_content=_CONTENT_SENTINEL,
            assistant_content="reply",
        )
        result = runner.invoke(
            app,
            [
                "show", "-j", "json-full", "--analyst-id", "alice",
                "--json", "--full",
            ],
        )
        assert result.exit_code == 0
        parsed = json.loads(result.stdout)
        assert any(item.get("content") == _CONTENT_SENTINEL for item in parsed)

    def test_last_n_limits_record_count(
        self, runner, env_with_key_and_db, db_path, fernet_key
    ):
        _seed_records(
            db_path, fernet_key,
            [(f"conv-{i}", f"content-{i}") for i in range(5)],
        )
        result = runner.invoke(
            app,
            [
                "show", "-j", "last-test", "--analyst-id", "alice",
                "--last", "2", "--json",
            ],
        )
        assert result.exit_code == 0
        parsed = json.loads(result.stdout)
        assert len(parsed) == 2

    def test_since_with_naive_datetime_exits_2_with_usage_error(
        self, runner, env_with_key_and_db, db_path, fernet_key
    ):
        _seed_pair(db_path, fernet_key)
        result = runner.invoke(
            app,
            [
                "show", "-j", "since-naive", "--analyst-id", "alice",
                "--since", "2026-01-01T00:00:00",
            ],
        )
        assert result.exit_code == 2
        assert "timezone" in result.stderr.lower()

    def test_until_with_naive_datetime_exits_2_with_usage_error(
        self, runner, env_with_key_and_db, db_path, fernet_key
    ):
        _seed_pair(db_path, fernet_key)
        result = runner.invoke(
            app,
            [
                "show", "-j", "until-naive", "--analyst-id", "alice",
                "--until", "2026-01-01T00:00:00",
            ],
        )
        assert result.exit_code == 2
        assert "timezone" in result.stderr.lower()

    def test_conversation_id_filter_returns_only_matching_records(
        self, runner, env_with_key_and_db, db_path, fernet_key
    ):
        _seed_records(
            db_path, fernet_key,
            [
                ("conv-A", "a1"),
                ("conv-B", "b1"),
                ("conv-A", "a2"),
                ("conv-B", "b2"),
            ],
        )
        result = runner.invoke(
            app,
            [
                "show", "-j", "filter-test", "--analyst-id", "alice",
                "--conversation-id", "conv-A", "--json",
            ],
        )
        assert result.exit_code == 0
        parsed = json.loads(result.stdout)
        assert len(parsed) == 2
        assert all(item["conversation_id"] == "conv-A" for item in parsed)

    def test_missing_justification_exits_2_usage_error(
        self, runner, env_with_key_and_db, db_path, fernet_key
    ):
        _seed_pair(db_path, fernet_key)
        result = runner.invoke(app, ["show", "--analyst-id", "alice"])
        assert result.exit_code == 2
        assert "justification" in result.stderr.lower()


# -----------------------------------------------------------------------------
# TestImport
# -----------------------------------------------------------------------------


class TestImport:
    """Happy path, provenance metadata, malformed-export rejection, flags."""

    def test_happy_path_writes_correct_number_of_records_and_exits_0(
        self, runner, env_with_key_and_db, db_path, fernet_key, tmp_path
    ):
        conv = _make_conv()  # 2 messages (1 human, 1 assistant)
        export = _make_export_file(tmp_path, [conv])
        result = runner.invoke(app, ["import", str(export)])
        assert result.exit_code == 0
        audit_count, _ = _count_records(db_path)
        assert audit_count == 2
        assert "imported 2 message(s) across 1 conversation(s)" in result.stdout

    def test_imported_records_have_source_import_request_id_none_detection_skipped(
        self, runner, env_with_key_and_db, db_path, fernet_key, tmp_path
    ):
        conv = _make_conv()
        export = _make_export_file(tmp_path, [conv])
        result = runner.invoke(app, ["import", str(export)])
        assert result.exit_code == 0
        records = _read_audit_records(db_path, fernet_key)
        for r in records:
            assert r.source == "import"
            assert r.request_id is None
            assert r.detection_status == "skipped"

    def test_imported_records_metadata_carries_asserted_provenance(
        self, runner, env_with_key_and_db, db_path, fernet_key, tmp_path
    ):
        conv = _make_conv(uuid="conv-prov-1")
        export = _make_export_file(
            tmp_path, [conv], filename="provenance-export.json"
        )
        result = runner.invoke(
            app, ["import", str(export), "--operator", "alice"]
        )
        assert result.exit_code == 0
        records = _read_audit_records(db_path, fernet_key)
        assert len(records) == 2
        for r in records:
            meta = r.metadata
            assert meta["imported_by"] == "alice"
            assert meta["source_file"] == "provenance-export.json"
            assert meta["original_message_uuid"] in {"msg-1", "msg-2"}
            assert meta["original_created_at"].startswith("2026-06-01T")

    def test_imported_records_conversation_id_equals_source_conversation_uuid(
        self, runner, env_with_key_and_db, db_path, fernet_key, tmp_path
    ):
        # Pins the field that groups imported messages by conversation,
        # and the one most likely to be silently wrong if the mapping
        # has a bug. Two conversations with distinct uuids.
        conv_a = _make_conv(uuid="conv-uuid-AAAA")
        conv_b = _make_conv(uuid="conv-uuid-BBBB")
        export = _make_export_file(tmp_path, [conv_a, conv_b])
        result = runner.invoke(app, ["import", str(export)])
        assert result.exit_code == 0
        records = _read_audit_records(db_path, fernet_key)
        assert len(records) == 4
        conv_ids = {r.conversation_id for r in records}
        assert conv_ids == {"conv-uuid-AAAA", "conv-uuid-BBBB"}

    def test_human_sender_maps_to_user_role_assistant_to_assistant(
        self, runner, env_with_key_and_db, db_path, fernet_key, tmp_path
    ):
        conv = _make_conv(messages=[
            {
                "uuid": "m-h",
                "text": "from human",
                "sender": "human",
                "created_at": "2026-06-01T10:00:00Z",
            },
            {
                "uuid": "m-a",
                "text": "from assistant",
                "sender": "assistant",
                "created_at": "2026-06-01T10:00:01Z",
            },
        ])
        export = _make_export_file(tmp_path, [conv])
        result = runner.invoke(app, ["import", str(export)])
        assert result.exit_code == 0
        records = _read_audit_records(db_path, fernet_key)
        by_text = {r.content: r.role for r in records}
        assert by_text["from human"] == "user"
        assert by_text["from assistant"] == "assistant"

    def test_malformed_json_aborts_phase_1_no_records_written(
        self, runner, env_with_key_and_db, db_path, fernet_key, tmp_path
    ):
        export = _write_raw_file(tmp_path, "not valid json {")
        result = runner.invoke(app, ["import", str(export)])
        assert result.exit_code == 2
        assert "not valid JSON" in result.stderr
        audit_count, _ = _count_records(db_path)
        assert audit_count == 0

    def test_missing_conversation_uuid_aborts_phase_1_no_records_written(
        self, runner, env_with_key_and_db, db_path, fernet_key, tmp_path
    ):
        bad_conv = {
            # uuid missing
            "chat_messages": [
                {
                    "uuid": "m1", "text": "x", "sender": "human",
                    "created_at": "2026-06-01T10:00:00Z",
                },
            ],
        }
        export = _make_export_file(tmp_path, [bad_conv])
        result = runner.invoke(app, ["import", str(export)])
        assert result.exit_code == 2
        assert "missing or invalid 'uuid'" in result.stderr
        audit_count, _ = _count_records(db_path)
        assert audit_count == 0

    def test_bad_sender_value_aborts_phase_1_no_records_written(
        self, runner, env_with_key_and_db, db_path, fernet_key, tmp_path
    ):
        conv = _make_conv(messages=[
            {
                "uuid": "m-bad",
                "text": "from robot",
                "sender": "robot",
                "created_at": "2026-06-01T10:00:00Z",
            },
        ])
        export = _make_export_file(tmp_path, [conv])
        result = runner.invoke(app, ["import", str(export)])
        assert result.exit_code == 2
        assert "sender must be 'human' or 'assistant'" in result.stderr
        audit_count, _ = _count_records(db_path)
        assert audit_count == 0

    def test_model_flag_lands_on_records(
        self, runner, env_with_key_and_db, db_path, fernet_key, tmp_path
    ):
        conv = _make_conv()
        export = _make_export_file(tmp_path, [conv])
        result = runner.invoke(
            app,
            ["import", str(export), "--model", "claude-3-5-sonnet-20241022"],
        )
        assert result.exit_code == 0
        records = _read_audit_records(db_path, fernet_key)
        for r in records:
            assert r.model == "claude-3-5-sonnet-20241022"

    def test_default_model_is_claude_import_sentinel(
        self, runner, env_with_key_and_db, db_path, fernet_key, tmp_path
    ):
        conv = _make_conv()
        export = _make_export_file(tmp_path, [conv])
        result = runner.invoke(app, ["import", str(export)])
        assert result.exit_code == 0
        records = _read_audit_records(db_path, fernet_key)
        for r in records:
            assert r.model == "claude-import"

    @pytest.mark.parametrize(
        "flag_value,env_value,expected",
        [
            (None, None, "unknown"),
            (None, "carol", "carol"),
            ("alice", "carol", "alice"),
        ],
        ids=["no-flag-no-env", "env-only", "flag-overrides-env"],
    )
    def test_operator_resolution_order_flag_then_env_then_unknown(
        self,
        runner,
        env_with_key_and_db,
        db_path,
        fernet_key,
        tmp_path,
        monkeypatch,
        flag_value,
        env_value,
        expected,
    ):
        if env_value is not None:
            monkeypatch.setenv(ENV_ANALYST_ID, env_value)
        conv = _make_conv()
        export = _make_export_file(tmp_path, [conv])
        args = ["import", str(export)]
        if flag_value is not None:
            args += ["--operator", flag_value]
        result = runner.invoke(app, args)
        assert result.exit_code == 0
        records = _read_audit_records(db_path, fernet_key)
        for r in records:
            assert r.metadata["imported_by"] == expected


# -----------------------------------------------------------------------------
# TestCrossCutting
# -----------------------------------------------------------------------------


class TestCrossCutting:
    """Properties that span multiple commands."""

    def test_config_error_message_does_not_echo_bad_key_value(
        self, runner, monkeypatch, db_path
    ):
        # The most security-critical CLI property. A malformed key
        # value must never appear in any user-visible error path. We
        # set a clearly-bad value with a distinctive sentinel and
        # confirm it does not bleed into stderr.
        sentinel = "NOT-A-REAL-KEY-DEADBEEF-DISTINCTIVE-SENTINEL"
        monkeypatch.setenv(ENV_ENCRYPTION_KEY, sentinel)
        monkeypatch.setenv(ENV_DB_PATH, str(db_path))
        result = runner.invoke(
            app, ["verify", "-j", "test", "--analyst-id", "alice"]
        )
        assert result.exit_code == 2
        # The bad value never appears in stderr or stdout.
        assert sentinel not in result.stderr
        assert sentinel not in result.stdout
        # The error message identifies WHAT failed without exposing the value.
        assert ENV_ENCRYPTION_KEY in result.stderr

    @pytest.mark.parametrize(
        "command_args",
        [
            ["verify", "-j", "x", "--analyst-id", "alice"],
            ["show", "-j", "x", "--analyst-id", "alice"],
            ["import", "PLACEHOLDER_PATH"],
        ],
        ids=["verify", "show", "import"],
    )
    def test_missing_key_exits_2_consistently_across_config_loading_commands(
        self, runner, tmp_path, command_args
    ):
        # No env at all: every Config-loading command must exit 2 with
        # a message naming LLMDR_ENCRYPTION_KEY.
        if command_args[0] == "import":
            # import needs a real path arg to get past typer's argument
            # validation and reach the config-loading code.
            export_path = tmp_path / "stub.json"
            export_path.write_text("[]", encoding="utf-8")
            command_args = command_args.copy()
            command_args[command_args.index("PLACEHOLDER_PATH")] = str(export_path)
        result = runner.invoke(app, command_args)
        assert result.exit_code == 2
        assert ENV_ENCRYPTION_KEY in result.stderr

    def test_justification_env_var_does_not_satisfy_required_flag_on_verify(
        self, runner, env_with_key_and_db, db_path, fernet_key, monkeypatch
    ):
        # Even with every plausibly relevant env var set, missing -j
        # is still a usage error. No env path bypasses the forensic
        # split for verify.
        _seed_pair(db_path, fernet_key)
        monkeypatch.setenv(ENV_ANALYST_ID, "alice")
        monkeypatch.setenv("LLMDR_JUSTIFICATION", "this-is-ignored")
        result = runner.invoke(app, ["verify"])
        assert result.exit_code == 2
        assert "justification" in result.stderr.lower()

    def test_justification_env_var_does_not_satisfy_required_flag_on_show(
        self, runner, env_with_key_and_db, db_path, fernet_key, monkeypatch
    ):
        _seed_pair(db_path, fernet_key)
        monkeypatch.setenv(ENV_ANALYST_ID, "alice")
        monkeypatch.setenv("LLMDR_JUSTIFICATION", "this-is-ignored")
        result = runner.invoke(app, ["show"])
        assert result.exit_code == 2
        assert "justification" in result.stderr.lower()

    def test_root_help_lists_all_five_commands(self, runner):
        # Catches accidental command unregistration in a future refactor.
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        for cmd in ("keygen", "serve", "verify", "show", "import"):
            assert cmd in result.output
