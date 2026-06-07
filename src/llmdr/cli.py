"""Command-line interface for llmdr.

Five subcommands cover the v0.1 operator surface:

* ``keygen`` mints a Fernet encryption key and prints it to stdout. No
  Config needed; the key is generated, not loaded.
* ``serve`` starts the FastAPI proxy from :func:`llmdr.proxy.create_app`
  bound to ``config.proxy_host`` and ``config.proxy_port``. Pure env-driven.
* ``verify`` runs :meth:`AuditStore.verify_chain` and exits 0 on intact,
  1 on tamper detection, 2 on configuration or usage error.
* ``show`` reads recent audit records. Defaults to metadata only; ``--full``
  opts into decrypted content. Every invocation writes one access_log
  entry. Justification is required on every call.
* ``import`` ingests a Claude conversation export into the audit store as
  ``source="import"`` records.

A sixth command, ``report``, will land alongside ``report.py``.

Cross-cutting contracts (do not soften without an explicit decision):

* Config is loaded once at command entry via :func:`_load_config_or_die`.
  A :class:`ConfigError` becomes "message to stderr, exit 2"; the bad
  value is never echoed because Config.from_env already strips it from
  its error messages.
* ``analyst_id`` resolves in this order: ``--analyst-id`` flag,
  ``LLMDR_ANALYST_ID`` env (via ``config.default_analyst_id``), else
  exit 2 for read commands.
* ``justification`` is NEVER read from environment. It is a required
  flag on every read command (``verify``, ``show``). Required-flag
  enforcement is typer's job; the absence of any env fallback is ours.
* The encryption key is never logged or echoed anywhere by this module.
  ``keygen`` writes it to stdout once, by design; nothing else here may
  touch it as a logged value.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Optional

import typer
import uvicorn

from llmdr.audit import AuditRecord, AuditStore
from llmdr.config import Config, ConfigError, generate_encryption_key
from llmdr.proxy import create_app

logger = logging.getLogger(__name__)


EXIT_OK = 0
EXIT_TAMPER = 1
EXIT_USAGE = 2

# Loopback identifiers used to gate the serve startup warning. A bind
# outside this set surfaces a stderr warning before uvicorn starts.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


app = typer.Typer(
    name="llmdr",
    help="LLM Detection and Response: tamper-evident audit for LLM traffic.",
    no_args_is_help=True,
    add_completion=False,
)


# -----------------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------------


def _load_config_or_die() -> Config:
    """Load Config from env or exit 2 with the sanitized error message.

    ConfigError messages from config.py never include the bad value, so
    forwarding the message to stderr is safe by construction.
    """
    try:
        return Config.from_env()
    except ConfigError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=EXIT_USAGE)


def _configure_logging(level: str) -> None:
    """Configure root logging so llmdr.* loggers emit through one sink."""
    # getattr(logging, level) is safe because config.py validates
    # log_level against exactly the set of logging module level-name
    # attributes (DEBUG, INFO, WARNING, ERROR, CRITICAL); any other
    # value would have already raised ConfigError at load time.
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )


def _resolve_analyst_id_or_die(
    flag_value: Optional[str], config: Config
) -> str:
    """Apply the flag-then-env resolution order; exit 2 if neither set.

    Used by read commands (verify, show). Write commands have their own
    fallback rules because writes do not produce access_log entries and
    can tolerate a less strict "operator unknown" default.
    """
    resolved = flag_value or config.default_analyst_id
    if not resolved:
        typer.echo(
            "analyst-id required: set LLMDR_ANALYST_ID or pass --analyst-id.",
            err=True,
        )
        raise typer.Exit(code=EXIT_USAGE)
    return resolved


def _row_counts(db_path: Path) -> tuple[int, int]:
    """Return (audit_records_count, access_log_count).

    Opens its own short-lived sqlite3 connection so the count read does
    not interleave with any AuditStore transaction state and does not
    write an access_log entry. Read-only metadata, no decryption.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        audit_count = conn.execute(
            "SELECT COUNT(*) FROM audit_records"
        ).fetchone()[0]
        access_count = conn.execute(
            "SELECT COUNT(*) FROM access_log"
        ).fetchone()[0]
        return int(audit_count), int(access_count)
    finally:
        conn.close()


def _last_access_log_id(db_path: Path) -> int:
    """Return the highest access_log id, used to pin the just-written entry.

    Assumes no concurrent writer between the get_records call and this
    query. The CLI is a single-operator interactive surface in v0.1, so
    this assumption holds; if multi-writer scenarios become real, this
    helper is the right place to revisit.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute("SELECT MAX(id) FROM access_log").fetchone()
        return int(row[0]) if row[0] is not None else 0
    finally:
        conn.close()


def _parse_iso8601_aware(
    value: Optional[str], *, flag_name: str
) -> Optional[datetime]:
    """Parse an ISO8601 string and reject naive datetimes.

    audit.py rejects naive datetimes at its public boundary, so we
    enforce timezone-aware input here and surface the failure as a
    typer usage error rather than letting it raise from inside the
    store call.
    """
    if value is None:
        return None
    # datetime.fromisoformat in Python 3.11+ accepts the "Z" suffix.
    try:
        dt = datetime.fromisoformat(value)
    except ValueError as exc:
        raise typer.BadParameter(
            f"could not parse ISO8601 datetime for {flag_name}: {value!r} ({exc})"
        )
    if dt.tzinfo is None:
        raise typer.BadParameter(
            f"{flag_name} must include timezone (e.g. '...Z' or '+00:00'): {value!r}"
        )
    return dt


# -----------------------------------------------------------------------------
# keygen
# -----------------------------------------------------------------------------


@app.command()
def keygen(
    quiet: Annotated[
        bool,
        typer.Option(
            "--quiet", "-q",
            help="Suppress the stderr handling warning; print the key only.",
        ),
    ] = False,
) -> None:
    """Generate a new Fernet encryption key and print it to stdout.

    The key is written to stdout as a single line with no prefix, so
    pipeline capture (e.g. piping into a secret manager) is clean. A
    multi-line handling warning is written to stderr unless --quiet.

    The key is never persisted to disk or logged by this command. The
    one path it takes out of the process is stdout.
    """
    key = generate_encryption_key()
    if not quiet:
        warning = (
            "Generated a new Fernet encryption key (printed to stdout).\n"
            "\n"
            "Handle this key like a password:\n"
            "  * Prefer piping into a secret manager or writing into an env\n"
            "    file (chmod 600) over capturing into a shell variable. Shell\n"
            "    variables leak to child processes and to 'set' or 'env' dumps,\n"
            "    which is one of the most common ways secrets escape.\n"
            "  * Add it to your environment as LLMDR_ENCRYPTION_KEY.\n"
            "  * Never commit it to version control.\n"
            "  * If you lose it, the audit store encrypted with it is\n"
            "    unrecoverable.\n"
            "  * It is now in your terminal scrollback; clear scrollback if\n"
            "    appropriate.\n"
            "\n"
            "Suppress this notice with --quiet."
        )
        typer.echo(warning, err=True)
    # sys.stdout.write avoids any framework formatting; the contract is
    # exactly the key bytes followed by one newline, nothing else.
    sys.stdout.write(key.decode("ascii") + "\n")


# -----------------------------------------------------------------------------
# serve
# -----------------------------------------------------------------------------


@app.command()
def serve() -> None:
    """Start the FastAPI proxy bound to the configured host and port.

    Pure env-driven. To change host, port, upstream URL, or log level,
    set LLMDR_PROXY_HOST, LLMDR_PROXY_PORT, LLMDR_UPSTREAM_BASE_URL, or
    LLMDR_LOG_LEVEL respectively before invocation. There are no
    overrides on this command by design; a flag-vs-env precedence
    question is more surface than it is worth at v0.1.
    """
    config = _load_config_or_die()
    _configure_logging(config.log_level)

    if config.proxy_host not in _LOOPBACK_HOSTS:
        warning = (
            f"WARNING: llmdr is binding to {config.proxy_host}:{config.proxy_port} "
            f"(non-loopback).\n"
            "This exposes the proxy beyond the local machine. llmdr v0.1 has no\n"
            "client authentication; anyone who can reach this address can use\n"
            "any API key sent through the proxy and observe the request stream.\n"
            "Bind to 127.0.0.1 unless you understand this exposure."
        )
        typer.echo(warning, err=True)

    fastapi_app = create_app(config)
    uvicorn.run(
        fastapi_app,
        host=config.proxy_host,
        port=config.proxy_port,
        log_level=config.log_level.lower(),
    )


# -----------------------------------------------------------------------------
# verify
# -----------------------------------------------------------------------------


@app.command()
def verify(
    justification: Annotated[
        str,
        typer.Option(
            "--justification", "-j",
            help="Business justification for this verification. Required.",
        ),
    ],
    analyst_id: Annotated[
        Optional[str],
        typer.Option(
            "--analyst-id",
            help="Operator identity. Falls back to LLMDR_ANALYST_ID env.",
        ),
    ] = None,
) -> None:
    """Verify the audit chain integrity.

    Exits 0 if both chains are intact, 1 if tamper is detected, 2 on
    configuration or usage error. Always writes one access_log entry
    recording this verification.
    """
    config = _load_config_or_die()
    _configure_logging(config.log_level)
    resolved_analyst = _resolve_analyst_id_or_die(analyst_id, config)

    with AuditStore(config.db_path, config.encryption_key) as store:
        result = store.verify_chain(
            analyst_id=resolved_analyst,
            justification=justification,
        )

    # Counts are read after verify_chain so they reflect the post-verify
    # state of the store (the access_log count includes the entry just
    # written by verify_chain itself).
    audit_count, access_count = _row_counts(config.db_path)

    if result.ok:
        typer.echo("audit chain intact")
        typer.echo(f"  audit_records: {audit_count} rows")
        typer.echo(f"  access_log: {access_count} rows")
        raise typer.Exit(code=EXIT_OK)

    typer.echo("audit chain BROKEN")
    if result.audit_tampered_at is not None:
        typer.echo(
            f"  audit_records broken at id={result.audit_tampered_at}"
        )
    else:
        typer.echo(f"  audit_records: intact ({audit_count} rows)")
    if result.access_log_tampered_at is not None:
        typer.echo(
            f"  access_log broken at id={result.access_log_tampered_at}"
        )
    else:
        typer.echo(f"  access_log: intact ({access_count} rows)")
    raise typer.Exit(code=EXIT_TAMPER)


# -----------------------------------------------------------------------------
# show
# -----------------------------------------------------------------------------


@app.command()
def show(
    justification: Annotated[
        str,
        typer.Option(
            "--justification", "-j",
            help="Business justification for this read. Required.",
        ),
    ],
    last: Annotated[
        int,
        typer.Option(
            "--last", "-n",
            min=1,
            help="Return at most this many of the most recent records.",
        ),
    ] = 10,
    conversation_id: Annotated[
        Optional[str],
        typer.Option(
            "--conversation-id", "-c",
            help="Filter to a single conversation id.",
        ),
    ] = None,
    since: Annotated[
        Optional[str],
        typer.Option(
            "--since",
            help="Lower-bound ts, ISO8601 with timezone (e.g. 2026-06-01T00:00:00Z).",
        ),
    ] = None,
    until: Annotated[
        Optional[str],
        typer.Option(
            "--until",
            help="Upper-bound ts, ISO8601 with timezone.",
        ),
    ] = None,
    full: Annotated[
        bool,
        typer.Option(
            "--full", "-f",
            help="Show full decrypted content per record. Default omits content.",
        ),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Emit JSON instead of a human-readable table.",
        ),
    ] = False,
    analyst_id: Annotated[
        Optional[str],
        typer.Option(
            "--analyst-id",
            help="Operator identity. Falls back to LLMDR_ANALYST_ID env.",
        ),
    ] = None,
    case_ref: Annotated[
        Optional[str],
        typer.Option(
            "--case-ref",
            help="Optional case or ticket reference recorded on the access_log entry.",
        ),
    ] = None,
) -> None:
    """Read recent audit records.

    Default output is a metadata table with no decrypted content. Pass
    --full to include content. Every invocation writes one access_log
    entry whether content is displayed or not; that entry pins who
    asked, when, with what filters, and why.
    """
    config = _load_config_or_die()
    _configure_logging(config.log_level)
    resolved_analyst = _resolve_analyst_id_or_die(analyst_id, config)

    since_dt = _parse_iso8601_aware(since, flag_name="--since")
    until_dt = _parse_iso8601_aware(until, flag_name="--until")

    with AuditStore(config.db_path, config.encryption_key) as store:
        records = store.get_records(
            last=last,
            conversation_id=conversation_id,
            since=since_dt,
            until=until_dt,
            analyst_id=resolved_analyst,
            justification=justification,
            case_ref=case_ref,
        )

    if json_output:
        _emit_records_json(records, include_content=full)
    elif full:
        _emit_records_full(records)
    else:
        _emit_records_table(records)

    if full and records:
        # Nudge the operator about the access_log entry that records
        # this read. Not a lecture, just a pointer back to the trail.
        access_id = _last_access_log_id(config.db_path)
        typer.echo(
            f"Decrypted content displayed. The access_log records this "
            f"read with id={access_id}.",
            err=True,
        )


def _truncate(s: str, width: int) -> str:
    if len(s) <= width:
        return s.ljust(width)
    return s[: width - 2] + ".."


def _emit_records_table(records: list[AuditRecord]) -> None:
    if not records:
        typer.echo("(no records)")
        return
    typer.echo(
        f"{'id':>5}  {'ts':<27}  {'conv_id':<12}  {'role':<9}  "
        f"{'model':<22}  {'len':>6}  detection"
    )
    for r in records:
        typer.echo(
            f"{r.id:>5}  {r.ts:<27}  {_truncate(r.conversation_id, 12)}  "
            f"{r.role:<9}  {_truncate(r.model, 22)}  "
            f"{len(r.content):>6}  {r.detection_status}"
        )


def _emit_records_full(records: list[AuditRecord]) -> None:
    if not records:
        typer.echo("(no records)")
        return
    for r in records:
        typer.echo(
            f"[id={r.id} ts={r.ts} conv={r.conversation_id} "
            f"role={r.role} model={r.model} detection={r.detection_status}]"
        )
        typer.echo(r.content)
        typer.echo("")


def _emit_records_json(
    records: list[AuditRecord], *, include_content: bool
) -> None:
    out: list[dict[str, Any]] = []
    for r in records:
        item: dict[str, Any] = {
            "id": r.id,
            "ts": r.ts,
            "source": r.source,
            "conversation_id": r.conversation_id,
            "request_id": r.request_id,
            "role": r.role,
            "model": r.model,
            "metadata": r.metadata,
            "detection_status": r.detection_status,
            "key_version": r.key_version,
            "prev_hash": r.prev_hash,
            "row_hash": r.row_hash,
        }
        if include_content:
            item["content"] = r.content
        out.append(item)
    typer.echo(json.dumps(out, indent=2, ensure_ascii=False))


# -----------------------------------------------------------------------------
# import
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class _ImportedMessage:
    uuid: str
    text: str
    role: str  # normalized to "user" or "assistant"
    created_at: str


@dataclass(frozen=True)
class _ImportedConversation:
    uuid: str
    messages: list[_ImportedMessage]


class _ImportValidationError(Exception):
    """Raised by phase 1 validation; phase 2 never runs if this fires."""


@app.command(name="import")
def import_cmd(
    path: Annotated[
        Path,
        typer.Argument(
            help="Path to the Claude conversation export JSON file.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    model: Annotated[
        str,
        typer.Option(
            "--model",
            help=(
                "Model identifier recorded on every imported record. The "
                "default 'claude-import' marks records as imported and "
                "model-unknown."
            ),
        ),
    ] = "claude-import",
    operator: Annotated[
        Optional[str],
        typer.Option(
            "--operator",
            help=(
                "Operator identity recorded in metadata.imported_by. Falls "
                "back to LLMDR_ANALYST_ID env, then 'unknown'."
            ),
        ),
    ] = None,
) -> None:
    """Ingest a Claude conversation export into the audit store.

    Provenance distinction (read this before relying on imported
    records in an investigation):

      Imported records carry **asserted** provenance. The operator
      asserts that this content came from the named export file. The
      metadata fields ``source_file``, ``original_message_uuid``, and
      ``original_created_at`` are the only trail back to the source;
      there is no cryptographic link between an imported record and
      the conversation it claims to represent. The operator vouches
      for the chain of custody.

      Live proxy records (``source='live'``) carry **observed**
      provenance. llmdr saw the traffic itself as it flowed through
      the proxy, and the record commits to bytes llmdr observed.

    An investigator should weight these classes differently. The
    distinction is the central forensic difference between import and
    live records and is intentionally surfaced on every record via the
    ``source`` column.

    Validation is two-phase. Phase 1 parses and validates the entire
    export with no audit writes; any malformed structure aborts before
    any record reaches the store. Phase 2 writes records one at a time.
    A phase-2 failure (e.g. disk full) leaves the chain consistent for
    the records already written and reports the count loudly.
    """
    config = _load_config_or_die()
    _configure_logging(config.log_level)

    resolved_operator = operator or config.default_analyst_id or "unknown"

    # Phase 1: read, parse, validate. No audit writes occur here.
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        typer.echo(f"failed to read {path}: {exc}", err=True)
        raise typer.Exit(code=EXIT_USAGE)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        typer.echo(f"{path} is not valid JSON: {exc}", err=True)
        raise typer.Exit(code=EXIT_USAGE)

    conversations = _normalize_conversations(parsed)
    if conversations is None:
        typer.echo(
            f"{path} top-level must be a conversation object or array of "
            f"conversations.",
            err=True,
        )
        raise typer.Exit(code=EXIT_USAGE)

    try:
        plan = _validate_import_plan(conversations)
    except _ImportValidationError as exc:
        typer.echo(f"import validation failed: {exc}", err=True)
        raise typer.Exit(code=EXIT_USAGE)

    total_messages = sum(len(c.messages) for c in plan)
    source_file = path.name

    # Phase 2: write. Per-record transactions inside AuditStore;
    # a mid-stream failure leaves a consistent chain for the prefix
    # already written.
    written = 0
    try:
        with AuditStore(config.db_path, config.encryption_key) as store:
            for conv in plan:
                for msg in conv.messages:
                    metadata: dict[str, Any] = {
                        "imported_by": resolved_operator,
                        "source_file": source_file,
                        "original_message_uuid": msg.uuid,
                        "original_created_at": msg.created_at,
                    }
                    store.write_record(
                        source="import",
                        conversation_id=conv.uuid,
                        request_id=None,
                        role=msg.role,  # type: ignore[arg-type]
                        content=msg.text,
                        model=model,
                        metadata=metadata,
                        detection_status="skipped",
                    )
                    written += 1
    except Exception as exc:
        # Partial-failure output. Loud, exact count, explicit verify
        # recommendation. Partial import state must never be silent.
        typer.echo(
            f"\nimport FAILED after writing {written} of {total_messages} "
            f"records.\n"
            f"Error: {type(exc).__name__}: {exc}\n"
            f"The chain remains internally consistent for the {written} "
            f"records already written. Run 'llmdr verify' to confirm chain "
            f"integrity, then investigate before re-running import.",
            err=True,
        )
        raise typer.Exit(code=EXIT_USAGE)

    typer.echo(
        f"imported {written} message(s) across {len(plan)} conversation(s) "
        f"from {source_file}"
    )


def _normalize_conversations(parsed: object) -> Optional[list[dict[str, Any]]]:
    """Accept either a single conversation object or an array of them."""
    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list) and all(isinstance(item, dict) for item in parsed):
        return parsed  # type: ignore[return-value]
    return None


def _validate_import_plan(
    conversations: list[dict[str, Any]],
) -> list[_ImportedConversation]:
    """Validate every conversation and message; return a normalized plan.

    Raises :class:`_ImportValidationError` on the first malformed entry.
    Atomic by construction: no plan returned means no audit writes will
    happen.
    """
    plan: list[_ImportedConversation] = []
    for conv_idx, conv in enumerate(conversations):
        conv_uuid = conv.get("uuid")
        if not isinstance(conv_uuid, str) or not conv_uuid:
            raise _ImportValidationError(
                f"conversation #{conv_idx}: missing or invalid 'uuid'"
            )
        chat_messages = conv.get("chat_messages")
        if not isinstance(chat_messages, list):
            raise _ImportValidationError(
                f"conversation {conv_uuid!r}: missing or invalid "
                f"'chat_messages' array"
            )

        validated: list[_ImportedMessage] = []
        for msg_idx, msg in enumerate(chat_messages):
            if not isinstance(msg, dict):
                raise _ImportValidationError(
                    f"conversation {conv_uuid!r} message #{msg_idx}: not an object"
                )
            msg_uuid = msg.get("uuid")
            text = msg.get("text")
            sender = msg.get("sender")
            created_at = msg.get("created_at")
            if not isinstance(msg_uuid, str) or not msg_uuid:
                raise _ImportValidationError(
                    f"conversation {conv_uuid!r} message #{msg_idx}: "
                    f"missing or invalid 'uuid'"
                )
            # Empty text is allowed (a sent-but-empty message); other
            # required fields must be non-empty strings.
            if not isinstance(text, str):
                raise _ImportValidationError(
                    f"conversation {conv_uuid!r} message {msg_uuid!r}: "
                    f"missing or non-string 'text'"
                )
            if not isinstance(sender, str) or not sender:
                raise _ImportValidationError(
                    f"conversation {conv_uuid!r} message {msg_uuid!r}: "
                    f"missing or invalid 'sender'"
                )
            if not isinstance(created_at, str) or not created_at:
                raise _ImportValidationError(
                    f"conversation {conv_uuid!r} message {msg_uuid!r}: "
                    f"missing or invalid 'created_at'"
                )
            if sender == "human":
                role = "user"
            elif sender == "assistant":
                role = "assistant"
            else:
                raise _ImportValidationError(
                    f"conversation {conv_uuid!r} message {msg_uuid!r}: "
                    f"sender must be 'human' or 'assistant', got {sender!r}"
                )
            validated.append(
                _ImportedMessage(
                    uuid=msg_uuid,
                    text=text,
                    role=role,
                    created_at=created_at,
                )
            )
        plan.append(
            _ImportedConversation(uuid=conv_uuid, messages=validated)
        )
    return plan


if __name__ == "__main__":
    app()
