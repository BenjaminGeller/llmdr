# CLAUDE.md

This file provides project context for Claude Code when working in the llmdr repository.

## Project overview

**llmdr** (LLM Detection and Response) is an open source forensic audit proxy for LLM traffic. It captures prompt/response pairs from LLM API traffic into a tamper-evident audit trail suitable for forensic investigation, compliance review, and insider risk work.

The project is owned by Benjamin Geller. It is MIT licensed and developed publicly at https://github.com/BenjaminGeller/llmdr. The PyPI package is at https://pypi.org/project/llmdr/.

llmdr is a personal project. It is not affiliated with any employer. Do not introduce employer-specific references, branding, or assumptions into the code, comments, documentation, or commit messages.

## Worldview and design philosophy

llmdr exists because of a specific belief: users and organizations should own and control the forensic data trail of their AI interactions, with cryptographic guarantees of integrity, rather than depending on vendor-controlled telemetry that may not be accessible, complete, or trustworthy in an investigation.

This worldview drives several design decisions:

- **Forensic primitives come first.** Hash chained logs, content encryption at rest, integrity verification, access logging. These are not optional features. They are the reason the tool exists.
- **Observability only, never enforcement, in v0.1.** llmdr does not block, redact, or modify traffic. It records and detects. Active enforcement is out of scope and may always remain out of scope.
- **Investigator workflow first, dashboard workflow second.** Outputs are designed for someone reconstructing what happened, not for a SOC analyst monitoring a live feed.
- **Watching the watchers.** Every read of the audit store is itself logged with analyst identity, query parameters, and a business justification field. Access logging is non-negotiable.
- **Vendor abstracted, Claude first.** The architecture supports adding other LLM providers but v0.1 ships with Claude support only. Provider-specific code is isolated behind an interface.

## v0.1 scope (locked, do not expand)

The following is the complete v0.1 feature set. Do not add features outside this scope without an explicit decision in conversation with the user. Scope discipline is the primary risk to shipping.

**Ingestion (two modes):**

1. **Live proxy mode.** A FastAPI service that proxies the Anthropic Messages API endpoint. Client SDKs are pointed at localhost; llmdr forwards transparently to api.anthropic.com and writes an audit record per prompt/response pair.
2. **Import mode.** A CLI command that ingests Claude conversation export JSON into the same record format used by the live proxy.

**Storage:**

- SQLite database (single file) holding the audit store and access log.
- Hash chained audit records: `sha256(prev_hash || canonical_json(row_without_hash))`.
- Schema: `id, ts, source (live|import), conversation_id, role, content, model, metadata_json, prev_hash, row_hash`.
- Content columns encrypted at rest using Fernet, key from environment variable.
- Separate `access_log` table tracking every query into the audit store with analyst identity, query parameters, business justification, and case/ticket reference.

**Detection:**

- Microsoft Presidio for PII detection.
- detect-secrets for credential detection.
- Detection hits stored in metadata. v0.1 does not block, redact, or modify content based on detection.

**CLI commands:**

- `llmdr verify` — verify audit chain integrity, exit 0 on success, exit 1 on tamper detection, identify exact break location.
- `llmdr show --last N` — dump last N records.
- `llmdr report` — generate Jinja2 HTML investigator report with timeline, detector hits highlighted, chain verification badge, and summary statistics.
- `llmdr import <path>` — ingest Claude conversation export.

**Demo:**

- A `tamper_demo.py` script that creates 3 prompts, modifies a row directly in SQLite, then runs verify-chain to show the integrity break. This is the README money shot.

**Explicitly out of v0.1 scope:**

- Browser extension or browser-based capture.
- Policy engine, blocking, redaction, or any active enforcement.
- UEBA or behavioral scoring.
- SIEM forwarding.
- Multi-user authentication or admin console.
- OpenAI, Gemini, or other provider support (provider abstraction is built in but only Claude is wired up).
- Claude Code local JSONL ingestion (stretch goal, may slip to v0.2).
- Anthropic Compliance API integration (canonical v0.2 enterprise direction).

## File layout

```
llmdr/
├── pyproject.toml          # Package metadata, dependencies, console_scripts
├── README.md               # Public-facing documentation
├── LICENSE                 # MIT
├── CONTRIBUTING.md         # CLA reference and contribution guidelines
├── .gitignore
├── CLAUDE.md               # This file
├── src/
│   └── llmdr/
│       ├── __init__.py
│       ├── audit.py        # AuditStore class, hash chain logic, encryption
│       ├── proxy.py        # FastAPI proxy app
│       ├── cli.py          # Typer CLI commands
│       ├── config.py       # Configuration loading from env vars
│       ├── detectors.py    # Presidio and detect-secrets integration
│       └── report.py       # Jinja2 HTML report generation
├── tests/
│   ├── __init__.py
│   ├── test_audit.py       # Hash chain, encryption, access logging
│   ├── test_proxy.py       # Proxy passthrough, audit write
│   ├── test_cli.py         # Verify, show, report, import commands
│   └── test_detectors.py   # PII and secret detection accuracy
└── scripts/
    └── tamper_demo.py      # README money shot
```

## Coding conventions

**Language and version:**

- Python 3.10 or higher.
- Type hints on all public functions and methods.
- Use `from __future__ import annotations` at the top of every module to allow forward references and cleaner type hints.

**Style:**

- Follow PEP 8.
- Use Black for formatting (line length 100).
- Use Ruff for linting.
- Prefer explicit over clever. This is a forensic tool. Code clarity matters more than concision.

**Dependencies:**

- Keep the dependency tree minimal. Every dependency is something a reviewer or auditor will examine.
- Pin to compatible release versions (`~=`) in pyproject.toml, not exact pins.
- Required dependencies: fastapi, httpx, typer, jinja2, cryptography, presidio-analyzer, detect-secrets.
- Database: standard library sqlite3 only. No ORM. This is a deliberate choice for forensic transparency: every database operation in llmdr should be inspectable as plain SQL by a reviewer, with no abstraction layer hiding what touches the audit store. Do not introduce SQLAlchemy or any other ORM without an explicit decision in conversation with the user.
- Test dependencies: pytest, pytest-asyncio, httpx (for FastAPI test client).

**Logging:**

- Use the standard library `logging` module, not `print`.
- Every component gets its own logger via `logging.getLogger(__name__)`.
- Log at INFO for normal operation, DEBUG for detail, WARNING for recoverable issues, ERROR for failures.

**Error handling:**

- Never silently swallow exceptions. Either handle them explicitly or let them propagate.
- Hash chain integrity errors must be loud and explicit. They are the most important signal in the system.
- Never write a partial record to the audit store. Either the full record (with valid prev_hash and row_hash) is written, or nothing is written.

**Testing:**

- Every module in src/llmdr has a corresponding test_*.py file.
- Hash chain logic must have tests covering: empty store, single record, multiple records, intentionally tampered record, prev_hash mismatch.
- Aim for meaningful tests, not coverage percentage. A test that mocks everything and asserts nothing is worse than no test.

**Commit hygiene:**

- Commit messages: imperative mood, present tense, summary line under 70 chars, optional body for context.
- Group related changes into single commits.
- Do not commit debugging code, commented-out code, or TODO comments without an associated issue.

## Stylistic preferences for generated content

The user has a strong preference: **do not use em dashes (long dashes, the — character) anywhere** in code comments, docstrings, README content, commit messages, or any other generated text. Use commas, parentheses, semicolons, periods, or restructure the sentence instead. Em dashes read as an obvious AI tell and the user does not want them in any output.

This applies to all generated content in this repo, including documentation.

## Security and forensic constraints

These are non-negotiable, regardless of convenience:

- **No content ever logged to stdout or files outside the encrypted audit store.** If a developer needs to debug content, they unlock the store with the proper key. Plaintext content must never leak.
- **The encryption key is loaded from an environment variable, never hardcoded, never committed, never logged.** A leaked key means a leaked audit store. Treat key handling like password handling.
- **Hash chain integrity verification is the most important code path in the system.** It must be straightforward, well tested, and obviously correct on inspection. Do not optimize it cleverly. Do not introduce abstractions that obscure what it does.
- **Access logging is mandatory for every read.** There is no "trusted developer" mode that bypasses it. The user reading their own data still generates an access log entry.
- **No telemetry, analytics, or phone home of any kind.** llmdr does not call out to any external service except the LLM provider being proxied.

## Working style preferences

- The user prefers tight scope discipline. If a task seems to be expanding beyond v0.1 scope, stop and surface it for an explicit decision rather than building outside scope.
- The user has thought carefully about architecture and naming. Do not propose name changes for the project, modules, or commands without strong justification.
- The user values honest pushback over agreement. If a proposed approach has a real problem, say so directly.
- When making non-obvious decisions, leave a brief comment explaining the reasoning. The user reads code carefully and a one-line "why" comment saves a future round trip.

## Known related projects (for context, not for emulation)

- **llmdrive** is a sibling project by the same author, working name only, fully reserved on GitHub and PyPI. It addresses cross-LLM memory portability. Conceptually linked but technically separate. Do not import from or depend on llmdrive.
- **NVIDIA NemoClaw / OpenShell** are agent runtime security tools. They handle prevention and policy enforcement. llmdr is a complementary forensic evidence layer. The two categories are different and llmdr should not adopt NemoClaw patterns.
- **llm-d** is an unrelated Red Hat distributed inference project. The visual similarity to llmdr is acknowledged. Do not attempt to interoperate or borrow naming conventions.

## When to ask the user

Ask before:

- Adding any new dependency.
- Adding a feature outside the v0.1 scope above.
- Making changes to the hash chain logic or audit store schema.
- Changing the project name, module names, or CLI command names.
- Touching anything related to encryption key handling.

Do not ask before:

- Writing or modifying tests.
- Improving inline documentation.
- Refactoring within a single module that does not change behavior.
- Fixing obvious bugs caught by tests.
