"""Environment-based configuration loader.

Defines :class:`Config`, a frozen dataclass holding all process-level
settings, plus :class:`ConfigError`, raised when the environment is
missing required values or contains malformed ones.

Critical invariant: the encryption key is load-only. The loader reads
``LLMDR_ENCRYPTION_KEY`` from the environment, validates it as a
Fernet key, and stores it on the Config dataclass. There is no
fallback to a default key, no silent generation, no read from disk.
A missing or malformed key is a hard failure.

Key generation lives in a separate function
(:func:`generate_encryption_key`) that is never called from the load
path. Mixing those two operations would let a fresh install silently
produce a working AuditStore with a new key and an empty chain,
indistinguishable on the surface from a wiped store. That would be a
serious forensic flaw. The CLI's keygen command imports the generator
function directly; the loader does not.

The encryption key is never logged, never included in exception
messages, never echoed back to stdout or stderr.

Forensic separation of identity and justification:
    ``default_analyst_id`` is allowed to come from an env var because
    the analyst_id is a stable property of the operator (the "who"),
    and an env-level default is a reasonable convenience. The
    ``justification`` field, which records "why" a read happened, has
    no env fallback anywhere in the system; every read must supply a
    fresh justification at the point of invocation. That split is the
    forensic balance we want and must not be softened.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from urllib.parse import urlparse

from cryptography.fernet import Fernet

# Env var names. Single source of truth so tests and downstream
# consumers can reference these constants instead of string literals.
ENV_ENCRYPTION_KEY: Final = "LLMDR_ENCRYPTION_KEY"
ENV_DB_PATH: Final = "LLMDR_DB_PATH"
ENV_UPSTREAM_BASE_URL: Final = "LLMDR_UPSTREAM_BASE_URL"
ENV_PROXY_HOST: Final = "LLMDR_PROXY_HOST"
ENV_PROXY_PORT: Final = "LLMDR_PROXY_PORT"
ENV_LOG_LEVEL: Final = "LLMDR_LOG_LEVEL"
ENV_ANALYST_ID: Final = "LLMDR_ANALYST_ID"

# Defaults for optional fields.
DEFAULT_DB_PATH: Final = "./llmdr.db"
DEFAULT_UPSTREAM_BASE_URL: Final = "https://api.anthropic.com"
DEFAULT_PROXY_HOST: Final = "127.0.0.1"
DEFAULT_PROXY_PORT: Final = 8080
DEFAULT_LOG_LEVEL: Final = "INFO"

_VALID_LOG_LEVELS: Final = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
_VALID_URL_SCHEMES: Final = ("http", "https")


class ConfigError(Exception):
    """Raised when the environment is missing required values or
    contains malformed ones.

    Deliberately does NOT subclass AuditStoreError. Configuration
    problems and forensic audit problems are different domains; the
    exception hierarchy should not blur them. A user staring at a
    ConfigError should know it is a setup issue, not chain damage.
    """


def generate_encryption_key() -> bytes:
    """Generate a fresh Fernet key.

    Pure: no env reads, no disk writes, no logging. Intentionally
    separate from the config loader to keep the "generate" and "load"
    paths from ever blurring. The loader must NEVER call this function
    as a fallback; that would let a fresh install silently produce a
    working AuditStore with a new key and an empty chain, which would
    be indistinguishable from a wiped store on the surface.

    The CLI's keygen command imports this function and prints the
    result for the user to place into their environment. The key is
    never written to disk by llmdr code.

    Returns:
        A 44-byte urlsafe-base64-encoded Fernet key.
    """
    return Fernet.generate_key()


@dataclass(frozen=True)
class Config:
    """Process-level configuration loaded from environment variables.

    Construct via :meth:`Config.from_env`. Direct construction is
    allowed for tests, but production code should always come through
    the loader so validation runs.

    Attributes:
        encryption_key: Validated Fernet key as bytes. Plugs directly
            into AuditStore's ``encryption_key`` parameter; no
            conversion layer in between.
        db_path: Filesystem path to the SQLite audit store. Parent
            directory is verified to exist at load time so misconfig
            surfaces before the audit store is opened.
        upstream_base_url: LLM provider base URL with any trailing
            slash stripped, so proxy URL construction can append
            paths without producing double-slashes.
        proxy_host: Bind host for the FastAPI proxy. Defaults to
            loopback; binding to a non-loopback interface is an
            explicit opt-in by setting the env var.
        proxy_port: Bind port in [1, 65535].
        log_level: Standard logging level name, normalized to upper
            case.
        default_analyst_id: Optional fallback analyst identity for
            CLI commands. None when unset. Justification is never
            read from the environment and must always be supplied per
            invocation; see module docstring.
    """

    encryption_key: bytes
    db_path: Path
    upstream_base_url: str
    proxy_host: str
    proxy_port: int
    log_level: str
    default_analyst_id: str | None

    @classmethod
    def from_env(cls) -> Config:
        """Load and validate all fields from os.environ.

        Reads os.environ at call time with no module-level caching,
        so tests using monkeypatch.setenv / monkeypatch.delenv get a
        clean per-test environment without fighting any cached state.

        Raises:
            ConfigError: If LLMDR_ENCRYPTION_KEY is missing or
                malformed, if any optional field has an invalid
                value, or if LLMDR_DB_PATH points at a path whose
                parent directory does not exist.
        """
        return cls(
            encryption_key=_load_encryption_key(),
            db_path=_load_db_path(),
            upstream_base_url=_load_upstream_base_url(),
            proxy_host=_load_proxy_host(),
            proxy_port=_load_proxy_port(),
            log_level=_load_log_level(),
            default_analyst_id=_load_default_analyst_id(),
        )


def _load_encryption_key() -> bytes:
    """Load and validate the Fernet key from LLMDR_ENCRYPTION_KEY.

    The bad value is NEVER included in the raised exception, even on
    malformed input. A typo of a real key elsewhere should not bleed
    into a terminal scrollback through our error path.
    """
    value = os.environ.get(ENV_ENCRYPTION_KEY)
    if not value:
        raise ConfigError(
            f"{ENV_ENCRYPTION_KEY} is not set. "
            f"Generate one with 'llmdr keygen' and add it to your environment."
        )
    # We deliberately do not strip whitespace. A key with leading or
    # trailing whitespace is a configuration smell that should surface,
    # not silently get normalized away. Fernet's base64 decoder tolerates
    # trailing whitespace, so we have to reject it explicitly here to keep
    # the "exactly these bytes, nothing silently trimmed" property true.
    # The bad value is not echoed.
    if value != value.strip():
        raise ConfigError(
            f"{ENV_ENCRYPTION_KEY} is not a valid Fernet key "
            f"(value contains leading or trailing whitespace). "
            f"Generate a new one with 'llmdr keygen'."
        )
    key_bytes = value.encode("ascii", errors="replace")
    try:
        Fernet(key_bytes)
    except (ValueError, TypeError):
        # Fernet raises ValueError on malformed input. TypeError is
        # caught defensively in case a future cryptography version
        # changes what it raises. The bad value itself is never
        # included in the message.
        raise ConfigError(
            f"{ENV_ENCRYPTION_KEY} is not a valid Fernet key. "
            f"Generate a new one with 'llmdr keygen'."
        ) from None
    return key_bytes


def _load_db_path() -> Path:
    # We check that the parent directory exists, but we do NOT
    # pre-check whether it is writable. A read-only parent will
    # surface as a SQLite error when AuditStore opens the file. This
    # keeps config validation cheap and avoids a TOCTOU window between
    # the writability check and the actual open.
    raw = os.environ.get(ENV_DB_PATH) or DEFAULT_DB_PATH
    path = Path(raw).expanduser()
    parent = path.parent
    if not parent.exists():
        raise ConfigError(
            f"{ENV_DB_PATH} points at {path}, but the parent directory "
            f"{parent} does not exist. Create it before starting llmdr."
        )
    return path


def _load_upstream_base_url() -> str:
    raw = os.environ.get(ENV_UPSTREAM_BASE_URL) or DEFAULT_UPSTREAM_BASE_URL
    parsed = urlparse(raw)
    if parsed.scheme not in _VALID_URL_SCHEMES or not parsed.netloc:
        raise ConfigError(
            f"{ENV_UPSTREAM_BASE_URL} must be an http or https URL, got: {raw!r}."
        )
    # Strip trailing slash so proxy URL construction can append paths
    # cleanly. Reachability is intentionally NOT validated; tests and
    # mock servers point at non-Anthropic hosts legitimately.
    return raw.rstrip("/")


def _load_proxy_host() -> str:
    # Host validation is intentionally delegated to the uvicorn bind
    # step (unlike port, which we validate here as an integer range).
    # Robust host/IP validation is fiddly across IPv4, IPv6, and
    # hostnames, and uvicorn is the authoritative validator at bind
    # time. A bad host surfaces there with a clear error.
    return os.environ.get(ENV_PROXY_HOST) or DEFAULT_PROXY_HOST


def _load_proxy_port() -> int:
    raw = os.environ.get(ENV_PROXY_PORT)
    if not raw:
        return DEFAULT_PROXY_PORT
    try:
        port = int(raw)
    except ValueError:
        raise ConfigError(
            f"{ENV_PROXY_PORT} must be an integer between 1 and 65535, got: {raw!r}."
        ) from None
    if not 1 <= port <= 65535:
        raise ConfigError(
            f"{ENV_PROXY_PORT} must be an integer between 1 and 65535, got: {port}."
        )
    return port


def _load_log_level() -> str:
    raw = os.environ.get(ENV_LOG_LEVEL) or DEFAULT_LOG_LEVEL
    normalized = raw.upper()
    if normalized not in _VALID_LOG_LEVELS:
        raise ConfigError(
            f"{ENV_LOG_LEVEL} must be one of "
            f"{', '.join(_VALID_LOG_LEVELS)}, got: {raw!r}."
        )
    return normalized


def _load_default_analyst_id() -> str | None:
    """Load optional analyst identity from LLMDR_ANALYST_ID.

    Returns None when unset, which means CLI commands must require a
    --analyst-id flag. Justification is deliberately NOT loaded from
    the environment under any name; every read of the audit store
    must carry a fresh justification supplied at invocation time.
    """
    return os.environ.get(ENV_ANALYST_ID) or None
