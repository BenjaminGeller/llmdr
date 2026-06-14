# llmdr

Tamper-evident forensic audit for Anthropic (Claude) API traffic. llmdr captures every prompt and response that passes through it into a hash-chained, encrypted, independently verifiable audit trail built for investigation, compliance review, and insider-risk work.

Status: work in progress, v0.1. MIT licensed.

## What it is

When an AI interaction becomes the subject of an investigation, you are usually left trusting the provider's telemetry to be complete, accessible, and unaltered. llmdr exists so you do not have to. It sits between your client and the Anthropic Messages API, records each prompt/response pair into a local SQLite store, and chains every record so any later edit to that store is detectable and localizable.

llmdr observes and records. It does not block, redact, or modify traffic. Every read of the store is itself logged, so the audit trail also records who looked at it and why.

## See it catch a tamper

No setup. No key, no config, no network. The script builds a throwaway encrypted store in a temp directory, plays five acts, and deletes the store on exit.

```
python scripts/tamper_demo.py
```

Trimmed run:

```
Act 2: verify the fresh chain
  ok = True
  message: chain intact

Act 3: tamper with id=2 directly in sqlite
  Overwriting the stored content of audit_records id=2 via a raw
  sqlite connection, with llmdr not running.

Act 4: verify the tampered chain
  ok = False
  audit_records tampered_at = 2      (row_hash mismatch at id=2)
  access_log    tampered_at = None   (the independent chain is untouched)

Act 5: attacker recomputes id=2 row_hash to cover the edit
  ok = False
  audit_records tampered_at = 3      (prev_hash mismatch at id=3)
```

Detecting the edit is the easy part. Because each record commits to the previous record's hash, an attacker who edits one row and recomputes only that row's hash does not erase the break, they move it one row down, to the next record whose prev_hash still commits to the original value. A partial tamper is therefore always caught at the following row. Hiding the edit completely forces rewriting every record after it, so concealment is all-or-nothing across the entire tail, never a single-row patch. Independent per-row hashes would let an attacker repair just the row they touched; a hash chain does not.

llmdr does not prevent edits to the database file. It makes them detectable and names the row where the chain breaks.

## Install

Requires Python 3.10 or newer.

```
pip install llmdr
```

Or from source:

```
git clone https://github.com/BenjaminGeller/llmdr
cd llmdr
pip install -e .
```

## Quickstart

### Without an API key (import an existing conversation)

This path needs no network and no Anthropic credentials. It ingests a Claude conversation export into the audit store, then verifies and reads it back.

1. Generate an encryption key and put it in your environment. The key is printed to stdout; the handling notice goes to stderr.

```
export LLMDR_ENCRYPTION_KEY=$(llmdr keygen -q)
```

Handle this key like a password. If you lose it, the store encrypted with it is unrecoverable. It is never written to disk by llmdr.

2. Import a Claude conversation export (the conversations JSON from a Claude data export).

```
llmdr import ./conversations.json --operator alice
```

3. Verify the chain. Exit 0 means intact; exit 1 means a break was found and the exact row is reported.

```
llmdr verify --justification "post-import integrity check" --analyst-id alice
```

4. Read the records back. Every read writes one access_log entry recording who read, when, with what filters, and why.

```
llmdr show --last 10 --justification "reviewing imported conversation" --analyst-id alice
```

`show` prints metadata only by default; add `--full` to include decrypted content.

### With the live proxy

The proxy captures real traffic as it happens. It needs a real Anthropic API key, which your client continues to supply; llmdr forwards it upstream unchanged.

1. Set the encryption key as above, then start the proxy. It binds to 127.0.0.1:8080 by default.

```
export LLMDR_ENCRYPTION_KEY=$(llmdr keygen -q)
llmdr serve
```

2. Point an Anthropic client at the proxy instead of the API. Your API key and request are forwarded unchanged; the prompt and response are audited in passing.

```python
from anthropic import Anthropic

client = Anthropic(base_url="http://127.0.0.1:8080")
client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello"}],
)
```

Notes on the live path:

- Streaming is not supported in v0.1. Requests with `stream=true` are rejected with a 400.
- To group several calls under one conversation, send an `x-llmdr-conversation-id` header; otherwise each request gets a fresh id.
- If an audit write fails, the proxy returns 500 and withholds the response rather than deliver unaudited traffic. The forensic contract is absolute.

3. Verify and read with the same `llmdr verify` and `llmdr show` commands shown above.

Configuration is environment-driven: `LLMDR_DB_PATH` (default `./llmdr.db`), `LLMDR_UPSTREAM_BASE_URL` (default `https://api.anthropic.com`), `LLMDR_PROXY_HOST`, `LLMDR_PROXY_PORT`, `LLMDR_LOG_LEVEL`, and `LLMDR_ANALYST_ID` (a default operator identity for read commands).

## Forensic design

### Hash-chained records

Each audit record stores a `row_hash` over its own canonical contents, and that hash includes the previous record's hash. The records form a chain: changing any field of any row changes its hash, which breaks the link the next row depends on. `llmdr verify` walks the chain and reports the id of the first row that does not verify.

### Encryption at rest

Prompt and response content and per-record metadata are encrypted with Fernet before they touch disk. The key is loaded from `LLMDR_ENCRYPTION_KEY` and never persisted or logged by llmdr. The hash chain commits to the encrypted bytes, so integrity can be verified without decrypting anything and without the key.

### Watching the watchers

Reads are not free. Every `show` and every `verify` writes an entry to a separate, independently hash-chained `access_log` table: who read (analyst id), when, with what filters, and a required business justification. There is no bypass. The justification is never read from the environment; it must be supplied at the point of each read.

### The forensic contract

A record is written in full or not at all; there are no partial rows. In the live proxy, an audit write failure fails the client request rather than delivering unaudited traffic. Content never leaves the encrypted store: it is never written to logs, stdout, or any file outside the store.

## v0.1 scope

What llmdr does today:

- Live proxy for a single endpoint, `POST /v1/messages`, forwarding to the Anthropic Messages API.
- Import of Claude conversation exports into the same record format.
- Hash-chained, Fernet-encrypted SQLite audit store.
- Independently hash-chained access log for every read.
- `keygen`, `serve`, `verify`, `show`, and `import` commands.

What llmdr does not do (by design or not yet):

- No blocking, redaction, or modification of traffic. It records and detects only.
- No streaming. `stream=true` is rejected in v0.1.
- API traffic only. llmdr audits Claude usage routed through the proxy. It does not capture consumer web (claude.ai), desktop, or mobile app use of Claude; those clients talk to Anthropic directly over TLS and never pass through the proxy. The enterprise pattern is to block those consumer paths and route sanctioned use through the audited API.
- A single Anthropic endpoint only. The proxy currently targets the Anthropic Messages API directly, with OpenAI support planned for v0.2. The aim is to audit a controlled set of sanctioned tools (Claude, then OpenAI), not open-ended multi-provider support.
- PII and secret detection (Microsoft Presidio and detect-secrets) is planned and not yet implemented. Records currently carry a `skipped` detection status.
- The Jinja2 HTML investigator report is planned and not yet implemented.
- No multi-user authentication, no admin console, no SIEM forwarding, no behavioral scoring.

## Contributing

Contributions welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for details. We use a CLA to maintain licensing flexibility.
