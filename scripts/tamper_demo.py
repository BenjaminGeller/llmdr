"""Tamper-evidence demonstration for llmdr's hash-chained audit store.

This is the README money shot. It runs end to end with no setup:

  Act 1  Write three audit records as one short conversation.
  Act 2  Verify the chain. It is intact.
  Act 3  Edit one row's content directly in SQLite, bypassing llmdr
         entirely (an attacker with file access while llmdr is not
         running).
  Act 4  Verify again. The chain is broken, and verification names the
         exact row (id=2). The independent access_log chain is untouched,
         so it still verifies, showing the two-chain design.
  Act 5  The attacker tries to cover the edit by recomputing the tampered
         row's row_hash. Now id=2 verifies on its own, but the break simply
         moves to id=3: every later row's prev_hash still commits to the
         original id=2 hash. Covering one row forces rewriting the entire
         tail. That property is what makes this a hash chain rather than
         independent per-row hashing.

The script is also a regression guard. If any break is not caught at the
expected row, it exits non-zero, so CI notices if the chain is ever
weakened.

Key custody: the demo mints an ephemeral Fernet key, holds it only in
memory, and never writes it to disk, an env var, or stdout. That is why
verification runs in-process against AuditStore directly rather than
shelling out to the `llmdr verify` CLI, which would require putting the
key in the environment, exactly the exposure the project's custody
discipline forbids.
"""

from __future__ import annotations

import logging
import sqlite3
import sys
import tempfile
from pathlib import Path

from llmdr.audit import (
    AuditStore,
    VerifyResult,
    # Act 5 must produce a genuinely valid row_hash for the tampered row, so it
    # reuses audit.py's real hashing primitives rather than reimplementing them.
    # These names are module-private; importing them couples the demo to audit
    # internals, but it guarantees the recomputed hash is identical to what
    # verify_chain computes (same code path), and any refactor breaks this
    # import loudly instead of silently producing a wrong hash.
    _AUDIT_HASH_COLUMNS,
    _canonical_json,
    _sha256_hex,
)
from llmdr.config import generate_encryption_key

ANALYST_ID = "tamper-demo"
JUSTIFICATION = "tamper-evidence demonstration"
CONVERSATION_ID = "demo-conv-0001"
TAMPERED_CONTENT = "TAMPERED: ciphertext overwritten directly in sqlite"


def _section(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def _write_demo_records(store: AuditStore) -> None:
    # The content here is synthetic, authored by this demo. We still write it
    # with source="live" because the live proxy is llmdr's primary use case and
    # that is the path worth showing. This is a demonstration only: real use
    # must never write fabricated content as source="live". Live records are
    # meant to commit to traffic llmdr actually observed.
    records = [
        (
            "user",
            "req_demo_0001",
            "Summarize our Q3 acquisition targets and the price ceiling for each.",
        ),
        (
            "assistant",
            "req_demo_0002",
            "Top target: Northwind Logistics, ceiling $40M. Secondary: Acme Freight, ceiling $18M.",
        ),
        ("user", "req_demo_0003", "Understood. Begin outreach to Northwind first."),
    ]
    for role, request_id, content in records:
        record_id = store.write_record(
            source="live",
            conversation_id=CONVERSATION_ID,
            request_id=request_id,
            role=role,  # type: ignore[arg-type]
            content=content,
            model="claude-sonnet-4-5",
            metadata={"demo": True},
            detection_status="skipped",
        )
        preview = content if len(content) <= 60 else content[:57] + "..."
        print(f"  wrote id={record_id} role={role:<9} {preview}")


def _print_verify(result: VerifyResult) -> None:
    print(f"  ok = {result.ok}")
    print(f"  audit_records tampered_at = {result.audit_tampered_at}")
    print(f"  access_log    tampered_at = {result.access_log_tampered_at}")
    print(f"  message: {result.message}")


def _tamper_content(db_path: Path, record_id: int) -> None:
    # Raw connection, bypassing AuditStore entirely. This is the honest threat
    # model: someone with direct access to the database file while llmdr is not
    # running. We overwrite the hashed `content` column; verification recomputes
    # row_hash over the stored bytes, so this breaks the chain with no key.
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            "UPDATE audit_records SET content = ? WHERE id = ?",
            (TAMPERED_CONTENT, record_id),
        )
        conn.commit()
    finally:
        conn.close()


def _recompute_row_hash(db_path: Path, record_id: int) -> None:
    # The attacker recomputes the tampered row's row_hash using audit.py's real
    # hashing logic, so id=2 verifies on its own. The break does not vanish: it
    # moves to id=3, whose prev_hash still commits to the original id=2 hash.
    conn = sqlite3.connect(str(db_path))
    try:
        # Column names come from a trusted constant in audit.py, not user input.
        col_list = ", ".join(_AUDIT_HASH_COLUMNS)
        row = conn.execute(
            f"SELECT {col_list} FROM audit_records WHERE id = ?", (record_id,)
        ).fetchone()
        if row is None:
            raise RuntimeError(f"record id={record_id} not found")
        # Mirrors _verify_audit_records exactly: build the same dict, hash it the
        # same way. _canonical_json sorts keys, so this is order-independent.
        row_for_hash = dict(zip(_AUDIT_HASH_COLUMNS, row))
        new_hash = _sha256_hex(_canonical_json(row_for_hash))
        conn.execute(
            "UPDATE audit_records SET row_hash = ? WHERE id = ?",
            (new_hash, record_id),
        )
        conn.commit()
    finally:
        conn.close()


def main() -> int:
    # Presentation choice for a clean README capture only: silence audit.py's
    # verify-path error logging so it does not interleave with this script's
    # narrative. This does NOT reflect how verify behaves in real use. In a real
    # operation those logger.error lines (the row_hash / prev_hash mismatch
    # locations) are wanted, and an operator should see them. We suppress them
    # here, in the demo, never in audit.py. The name matches audit.py's
    # logging.getLogger(__name__) for module "llmdr.audit".
    logging.getLogger("llmdr.audit").setLevel(logging.CRITICAL)

    key = generate_encryption_key()  # ephemeral, in-memory only, never printed
    failures: list[str] = []

    def expect(condition: bool, description: str) -> None:
        status = "PASS" if condition else "FAIL"
        print(f"  [{status}] {description}")
        if not condition:
            failures.append(description)

    with tempfile.TemporaryDirectory(prefix="llmdr-tamper-demo-") as tmpdir:
        db_path = Path(tmpdir) / "demo.db"

        _section("Act 1: write three audit records")
        with AuditStore(db_path, key) as store:
            _write_demo_records(store)

        _section("Act 2: verify the fresh chain")
        with AuditStore(db_path, key) as store:
            result = store.verify_chain(analyst_id=ANALYST_ID, justification=JUSTIFICATION)
        _print_verify(result)
        expect(result.ok, "fresh chain verifies as intact")

        _section("Act 3: tamper with id=2 directly in sqlite")
        print("  Overwriting the stored content of audit_records id=2 via a raw")
        print("  sqlite connection, with llmdr not running.")
        _tamper_content(db_path, 2)

        _section("Act 4: verify the tampered chain")
        with AuditStore(db_path, key) as store:
            result = store.verify_chain(analyst_id=ANALYST_ID, justification=JUSTIFICATION)
        _print_verify(result)
        expect(not result.ok, "tampered chain verifies as BROKEN")
        expect(result.audit_tampered_at == 2, "break is reported at audit_records id=2")
        expect(
            result.access_log_tampered_at is None,
            "access_log chain stays intact (independent of audit_records)",
        )

        _section("Act 5: attacker recomputes id=2 row_hash to cover the edit")
        print("  Recomputing row_hash for id=2 with audit.py's real hashing, so")
        print("  id=2 verifies on its own. Watch the break move to id=3.")
        _recompute_row_hash(db_path, 2)
        with AuditStore(db_path, key) as store:
            result = store.verify_chain(analyst_id=ANALYST_ID, justification=JUSTIFICATION)
        _print_verify(result)
        expect(not result.ok, "chain is still BROKEN after the cover-up")
        expect(
            result.audit_tampered_at == 3,
            "break moved to id=3: covering id=2 forces rewriting the tail",
        )

        _section("Result")
        if failures:
            print(f"  DEMO FAILED: {len(failures)} expectation(s) not met:")
            for item in failures:
                print(f"    - {item}")
            return 1
        print("  All expectations met. The hash chain caught every tamper at the")
        print("  exact row, and covering a row forced the break down the chain.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
