"""Tests for the config loader.

Organized by what each test proves. The most security-critical class
is TestEncryptionKeyMalformed: it pins the property that a malformed
key value never appears in any user-visible exception surface.

* TestEncryptionKeyMissing
* TestEncryptionKeyMalformed (security-critical)
* TestEncryptionKeyNonAscii
* TestEncryptionKeyValid
* TestDbPath
* TestUpstreamBaseUrl
* TestProxyPort
* TestProxyHost
* TestLogLevel
* TestDefaultAnalystId
* TestFullFromEnv
* TestGenerateLoadSeparation
* TestConfigErrorHierarchy
* TestStructuralGuards
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from cryptography.fernet import Fernet

import llmdr.config
from llmdr.audit import AuditStoreError
from llmdr.config import (
    DEFAULT_DB_PATH,
    DEFAULT_LOG_LEVEL,
    DEFAULT_PROXY_HOST,
    DEFAULT_PROXY_PORT,
    DEFAULT_UPSTREAM_BASE_URL,
    ENV_ANALYST_ID,
    ENV_DB_PATH,
    ENV_ENCRYPTION_KEY,
    ENV_LOG_LEVEL,
    ENV_PROXY_HOST,
    ENV_PROXY_PORT,
    ENV_UPSTREAM_BASE_URL,
    Config,
    ConfigError,
    generate_encryption_key,
)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


ALL_ENV_VARS = (
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
    """Clear every LLMDR_* env var the loader looks at.

    Autouse so every test starts from a deterministic environment
    regardless of the host shell. Uses raising=False so a missing var
    is not an error.
    """
    for var in ALL_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def valid_key_str() -> str:
    """A fresh Fernet key as an ASCII string (the form env vars require)."""
    return Fernet.generate_key().decode("ascii")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _walk_chain(exc: BaseException) -> list[BaseException]:
    """Return exc plus every link in its cause/context chain."""
    seen: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and current not in seen:
        seen.append(current)
        # We follow both because str() of the top exception can surface
        # content from either __cause__ or __context__.
        current = current.__cause__ or current.__context__
    return seen


def _assert_sentinel_absent(exc: BaseException, sentinel: str) -> None:
    """Assert sentinel does not appear in str/repr/args of exc or any
    link in its cause/context chain."""
    for link in _walk_chain(exc):
        assert sentinel not in str(link), f"sentinel leaked into str({link!r})"
        assert sentinel not in repr(link), f"sentinel leaked into repr({link!r})"
        for arg in link.args:
            if isinstance(arg, str):
                assert sentinel not in arg, f"sentinel leaked into args: {arg!r}"
            else:
                assert sentinel not in repr(arg), (
                    f"sentinel leaked into args repr: {arg!r}"
                )


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


class TestEncryptionKeyMissing:
    """A missing key is a hard failure with a useful message. There is
    no value to leak in the missing case, but the message shape is
    pinned so it stays user-actionable."""

    def test_unset_raises_with_named_env_var_and_keygen_hint(self):
        with pytest.raises(ConfigError) as exc_info:
            Config.from_env()
        msg = str(exc_info.value)
        assert ENV_ENCRYPTION_KEY in msg
        assert "llmdr keygen" in msg

    def test_empty_string_treated_as_missing(self, monkeypatch):
        monkeypatch.setenv(ENV_ENCRYPTION_KEY, "")
        with pytest.raises(ConfigError) as exc_info:
            Config.from_env()
        msg = str(exc_info.value)
        assert ENV_ENCRYPTION_KEY in msg
        assert "llmdr keygen" in msg


class TestEncryptionKeyMalformed:
    """The bad key value never appears in any user-visible exception
    surface. Each test uses a distinctive sentinel string and asserts
    it is absent from str, repr, args, and the entire exception chain.
    This is the security-critical property of the module."""

    def test_garbage_non_base64_does_not_leak(self, monkeypatch):
        sentinel = "SENTINEL_GARBAGE_zzz_NOT_BASE64_!!!"
        monkeypatch.setenv(ENV_ENCRYPTION_KEY, sentinel)
        with pytest.raises(ConfigError) as exc_info:
            Config.from_env()
        _assert_sentinel_absent(exc_info.value, sentinel)
        assert "is not a valid Fernet key" in str(exc_info.value)
        assert ENV_ENCRYPTION_KEY in str(exc_info.value)

    def test_too_short_does_not_leak(self, monkeypatch):
        # 24 chars of valid urlsafe-base64 alphabet, decodes to 18
        # bytes (Fernet requires 32).
        sentinel = "SENTINEL_TOOSHORT_aabbcc"
        assert len(sentinel) == 24
        monkeypatch.setenv(ENV_ENCRYPTION_KEY, sentinel)
        with pytest.raises(ConfigError) as exc_info:
            Config.from_env()
        _assert_sentinel_absent(exc_info.value, sentinel)

    def test_wrong_length_does_not_leak(self, monkeypatch):
        # 44 chars of valid urlsafe-base64 alphabet, decodes to 33
        # bytes (Fernet requires exactly 32).
        sentinel = "SENTINEL_WRONGLEN_" + "A" * 26
        assert len(sentinel) == 44
        monkeypatch.setenv(ENV_ENCRYPTION_KEY, sentinel)
        with pytest.raises(ConfigError) as exc_info:
            Config.from_env()
        _assert_sentinel_absent(exc_info.value, sentinel)

    def test_whitespace_only_takes_malformed_path(self, monkeypatch):
        # Whitespace-only is truthy, so it passes the missing-check
        # and reaches the explicit whitespace check. This pins the
        # boundary: empty string is missing, but "   \t\n  " is
        # malformed.
        sentinel = "   \t\n  "
        monkeypatch.setenv(ENV_ENCRYPTION_KEY, sentinel)
        with pytest.raises(ConfigError) as exc_info:
            Config.from_env()
        msg = str(exc_info.value)
        # Malformed path message, not the missing-key message.
        assert "is not a valid Fernet key" in msg
        assert "is not set" not in msg
        _assert_sentinel_absent(exc_info.value, sentinel)

    def test_trailing_whitespace_on_valid_key_does_not_leak(
        self, monkeypatch, valid_key_str
    ):
        # A valid key with a trailing newline appended. Fernet's base64
        # decode would tolerate this silently, so the loader's explicit
        # whitespace check is what rejects it. The key body is the
        # sentinel here; assert it does not leak in the error message.
        bad_value = valid_key_str + "\n"
        monkeypatch.setenv(ENV_ENCRYPTION_KEY, bad_value)
        with pytest.raises(ConfigError) as exc_info:
            Config.from_env()
        msg = str(exc_info.value)
        assert "is not a valid Fernet key" in msg
        assert "whitespace" in msg
        _assert_sentinel_absent(exc_info.value, valid_key_str)

    def test_exception_chain_is_suppressed(self, monkeypatch):
        # `raise ... from None` should produce no __cause__, and the
        # sentinel must not surface via __context__ either.
        sentinel = "SENTINEL_CHAIN_CHECK_qqq"
        monkeypatch.setenv(ENV_ENCRYPTION_KEY, sentinel)
        with pytest.raises(ConfigError) as exc_info:
            Config.from_env()
        exc = exc_info.value
        assert exc.__cause__ is None
        _assert_sentinel_absent(exc, sentinel)


class TestEncryptionKeyNonAscii:
    """Non-ASCII content in the env var fails as ConfigError, never as
    UnicodeEncodeError. The non-ASCII character does not leak."""

    def test_smart_quote_raises_config_error_not_unicode_error(self, monkeypatch):
        sentinel = "SENTINEL_SMARTQ_xyz"
        smart_quote = "“"
        bad_value = f"{sentinel}{smart_quote}suffix_extra"
        monkeypatch.setenv(ENV_ENCRYPTION_KEY, bad_value)
        with pytest.raises(ConfigError) as exc_info:
            Config.from_env()
        exc = exc_info.value
        assert not isinstance(exc, UnicodeEncodeError)
        for link in _walk_chain(exc):
            assert not isinstance(link, UnicodeEncodeError)
        _assert_sentinel_absent(exc, sentinel)
        # The smart quote character itself does not leak either.
        for link in _walk_chain(exc):
            assert smart_quote not in str(link)


class TestEncryptionKeyValid:
    """A valid key loads to bytes Fernet actually accepts. The loader
    produces audit-store-compatible bytes with no transformation."""

    def test_round_trip_with_generated_key(self, monkeypatch):
        generated = generate_encryption_key()
        monkeypatch.setenv(ENV_ENCRYPTION_KEY, generated.decode("ascii"))
        config = Config.from_env()
        assert config.encryption_key == generated
        f = Fernet(config.encryption_key)
        plaintext = b"forensic round trip"
        assert f.decrypt(f.encrypt(plaintext)) == plaintext


class TestDbPath:
    """Defaults, validation, and tilde expansion for LLMDR_DB_PATH."""

    def test_default_when_unset(self, monkeypatch, valid_key_str):
        monkeypatch.setenv(ENV_ENCRYPTION_KEY, valid_key_str)
        assert Config.from_env().db_path == Path(DEFAULT_DB_PATH)

    def test_default_when_empty_string(self, monkeypatch, valid_key_str):
        monkeypatch.setenv(ENV_ENCRYPTION_KEY, valid_key_str)
        monkeypatch.setenv(ENV_DB_PATH, "")
        assert Config.from_env().db_path == Path(DEFAULT_DB_PATH)

    def test_valid_path_with_existing_parent(
        self, monkeypatch, valid_key_str, tmp_path
    ):
        target = tmp_path / "audit.db"
        monkeypatch.setenv(ENV_ENCRYPTION_KEY, valid_key_str)
        monkeypatch.setenv(ENV_DB_PATH, str(target))
        assert Config.from_env().db_path == target

    def test_nonexistent_parent_names_directory_in_error(
        self, monkeypatch, valid_key_str, tmp_path
    ):
        missing_parent = tmp_path / "does_not_exist"
        target = missing_parent / "audit.db"
        monkeypatch.setenv(ENV_ENCRYPTION_KEY, valid_key_str)
        monkeypatch.setenv(ENV_DB_PATH, str(target))
        with pytest.raises(ConfigError) as exc_info:
            Config.from_env()
        assert str(missing_parent) in str(exc_info.value)

    def test_tilde_expansion(self, monkeypatch, valid_key_str):
        monkeypatch.setenv(ENV_ENCRYPTION_KEY, valid_key_str)
        monkeypatch.setenv(ENV_DB_PATH, "~/llmdr_test_audit.db")
        config = Config.from_env()
        assert "~" not in str(config.db_path)
        assert str(config.db_path).startswith(str(Path.home()))


class TestUpstreamBaseUrl:
    """LLMDR_UPSTREAM_BASE_URL defaults, scheme validation, and
    trailing-slash stripping."""

    def test_default_when_unset(self, monkeypatch, valid_key_str):
        monkeypatch.setenv(ENV_ENCRYPTION_KEY, valid_key_str)
        assert Config.from_env().upstream_base_url == DEFAULT_UPSTREAM_BASE_URL

    def test_default_when_empty_string(self, monkeypatch, valid_key_str):
        monkeypatch.setenv(ENV_ENCRYPTION_KEY, valid_key_str)
        monkeypatch.setenv(ENV_UPSTREAM_BASE_URL, "")
        assert Config.from_env().upstream_base_url == DEFAULT_UPSTREAM_BASE_URL

    def test_http_url_accepted(self, monkeypatch, valid_key_str):
        monkeypatch.setenv(ENV_ENCRYPTION_KEY, valid_key_str)
        monkeypatch.setenv(ENV_UPSTREAM_BASE_URL, "http://localhost:8000")
        assert Config.from_env().upstream_base_url == "http://localhost:8000"

    def test_https_url_accepted(self, monkeypatch, valid_key_str):
        monkeypatch.setenv(ENV_ENCRYPTION_KEY, valid_key_str)
        monkeypatch.setenv(ENV_UPSTREAM_BASE_URL, "https://example.com")
        assert Config.from_env().upstream_base_url == "https://example.com"

    def test_trailing_slash_stripped(self, monkeypatch, valid_key_str):
        monkeypatch.setenv(ENV_ENCRYPTION_KEY, valid_key_str)
        monkeypatch.setenv(ENV_UPSTREAM_BASE_URL, "https://api.anthropic.com/")
        assert Config.from_env().upstream_base_url == "https://api.anthropic.com"

    def test_multiple_trailing_slashes_stripped(
        self, monkeypatch, valid_key_str
    ):
        monkeypatch.setenv(ENV_ENCRYPTION_KEY, valid_key_str)
        monkeypatch.setenv(ENV_UPSTREAM_BASE_URL, "https://api.anthropic.com///")
        assert Config.from_env().upstream_base_url == "https://api.anthropic.com"

    @pytest.mark.parametrize(
        "bad_url",
        ["ftp://example.com", "file:///etc/passwd", "https://", "not_a_url"],
    )
    def test_invalid_urls_rejected(self, monkeypatch, valid_key_str, bad_url):
        monkeypatch.setenv(ENV_ENCRYPTION_KEY, valid_key_str)
        monkeypatch.setenv(ENV_UPSTREAM_BASE_URL, bad_url)
        with pytest.raises(ConfigError) as exc_info:
            Config.from_env()
        assert ENV_UPSTREAM_BASE_URL in str(exc_info.value)


class TestProxyPort:
    """LLMDR_PROXY_PORT defaults, integer validation, and range."""

    def test_default_when_unset(self, monkeypatch, valid_key_str):
        monkeypatch.setenv(ENV_ENCRYPTION_KEY, valid_key_str)
        assert Config.from_env().proxy_port == DEFAULT_PROXY_PORT

    def test_default_when_empty_string(self, monkeypatch, valid_key_str):
        monkeypatch.setenv(ENV_ENCRYPTION_KEY, valid_key_str)
        monkeypatch.setenv(ENV_PROXY_PORT, "")
        assert Config.from_env().proxy_port == DEFAULT_PROXY_PORT

    def test_valid_mid_range_port(self, monkeypatch, valid_key_str):
        monkeypatch.setenv(ENV_ENCRYPTION_KEY, valid_key_str)
        monkeypatch.setenv(ENV_PROXY_PORT, "9090")
        assert Config.from_env().proxy_port == 9090

    def test_minimum_boundary_accepted(self, monkeypatch, valid_key_str):
        monkeypatch.setenv(ENV_ENCRYPTION_KEY, valid_key_str)
        monkeypatch.setenv(ENV_PROXY_PORT, "1")
        assert Config.from_env().proxy_port == 1

    def test_maximum_boundary_accepted(self, monkeypatch, valid_key_str):
        monkeypatch.setenv(ENV_ENCRYPTION_KEY, valid_key_str)
        monkeypatch.setenv(ENV_PROXY_PORT, "65535")
        assert Config.from_env().proxy_port == 65535

    @pytest.mark.parametrize("bad", ["0", "65536", "-1"])
    def test_out_of_range_rejected(self, monkeypatch, valid_key_str, bad):
        monkeypatch.setenv(ENV_ENCRYPTION_KEY, valid_key_str)
        monkeypatch.setenv(ENV_PROXY_PORT, bad)
        with pytest.raises(ConfigError) as exc_info:
            Config.from_env()
        assert ENV_PROXY_PORT in str(exc_info.value)

    def test_non_integer_rejected_with_value_in_message(
        self, monkeypatch, valid_key_str
    ):
        monkeypatch.setenv(ENV_ENCRYPTION_KEY, valid_key_str)
        monkeypatch.setenv(ENV_PROXY_PORT, "not_a_port")
        with pytest.raises(ConfigError) as exc_info:
            Config.from_env()
        msg = str(exc_info.value)
        assert ENV_PROXY_PORT in msg
        assert "not_a_port" in msg


class TestProxyHost:
    """LLMDR_PROXY_HOST: no validation, returned verbatim. Pins the
    documented asymmetry with proxy_port."""

    def test_default_when_unset(self, monkeypatch, valid_key_str):
        monkeypatch.setenv(ENV_ENCRYPTION_KEY, valid_key_str)
        assert Config.from_env().proxy_host == DEFAULT_PROXY_HOST

    def test_default_when_empty_string(self, monkeypatch, valid_key_str):
        monkeypatch.setenv(ENV_ENCRYPTION_KEY, valid_key_str)
        monkeypatch.setenv(ENV_PROXY_HOST, "")
        assert Config.from_env().proxy_host == DEFAULT_PROXY_HOST

    @pytest.mark.parametrize(
        "host",
        ["0.0.0.0", "::1", "my-host.local", "not_a_real_host", "10.0.0.1"],
    )
    def test_arbitrary_host_returned_verbatim(
        self, monkeypatch, valid_key_str, host
    ):
        monkeypatch.setenv(ENV_ENCRYPTION_KEY, valid_key_str)
        monkeypatch.setenv(ENV_PROXY_HOST, host)
        assert Config.from_env().proxy_host == host


class TestLogLevel:
    """LLMDR_LOG_LEVEL: defaults, case normalization, validation."""

    def test_default_when_unset(self, monkeypatch, valid_key_str):
        monkeypatch.setenv(ENV_ENCRYPTION_KEY, valid_key_str)
        assert Config.from_env().log_level == DEFAULT_LOG_LEVEL

    def test_default_when_empty_string(self, monkeypatch, valid_key_str):
        monkeypatch.setenv(ENV_ENCRYPTION_KEY, valid_key_str)
        monkeypatch.setenv(ENV_LOG_LEVEL, "")
        assert Config.from_env().log_level == DEFAULT_LOG_LEVEL

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("debug", "DEBUG"),
            ("Warning", "WARNING"),
            ("ERROR", "ERROR"),
            ("critical", "CRITICAL"),
            ("info", "INFO"),
        ],
    )
    def test_case_normalized(self, monkeypatch, valid_key_str, raw, expected):
        monkeypatch.setenv(ENV_ENCRYPTION_KEY, valid_key_str)
        monkeypatch.setenv(ENV_LOG_LEVEL, raw)
        assert Config.from_env().log_level == expected

    def test_invalid_level_rejected(self, monkeypatch, valid_key_str):
        monkeypatch.setenv(ENV_ENCRYPTION_KEY, valid_key_str)
        monkeypatch.setenv(ENV_LOG_LEVEL, "verbose")
        with pytest.raises(ConfigError) as exc_info:
            Config.from_env()
        msg = str(exc_info.value)
        assert ENV_LOG_LEVEL in msg
        for level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            assert level in msg


class TestDefaultAnalystId:
    """LLMDR_ANALYST_ID: optional, no validation. None when unset or
    empty."""

    def test_none_when_unset(self, monkeypatch, valid_key_str):
        monkeypatch.setenv(ENV_ENCRYPTION_KEY, valid_key_str)
        assert Config.from_env().default_analyst_id is None

    def test_none_when_empty_string(self, monkeypatch, valid_key_str):
        monkeypatch.setenv(ENV_ENCRYPTION_KEY, valid_key_str)
        monkeypatch.setenv(ENV_ANALYST_ID, "")
        assert Config.from_env().default_analyst_id is None

    @pytest.mark.parametrize(
        "value",
        ["analyst-1", "Alice Smith", "user@example.com", "опера тор"],
    )
    def test_value_returned_verbatim(
        self, monkeypatch, valid_key_str, value
    ):
        monkeypatch.setenv(ENV_ENCRYPTION_KEY, valid_key_str)
        monkeypatch.setenv(ENV_ANALYST_ID, value)
        assert Config.from_env().default_analyst_id == value


class TestFullFromEnv:
    """A fully populated valid environment produces a Config with all
    expected field values and types. The dataclass is frozen."""

    def test_full_round_trip_pins_values_and_types(
        self, monkeypatch, tmp_path
    ):
        key = Fernet.generate_key()
        db_target = tmp_path / "audit.db"
        monkeypatch.setenv(ENV_ENCRYPTION_KEY, key.decode("ascii"))
        monkeypatch.setenv(ENV_DB_PATH, str(db_target))
        monkeypatch.setenv(ENV_UPSTREAM_BASE_URL, "https://example.com/")
        monkeypatch.setenv(ENV_PROXY_HOST, "0.0.0.0")
        monkeypatch.setenv(ENV_PROXY_PORT, "9090")
        monkeypatch.setenv(ENV_LOG_LEVEL, "debug")
        monkeypatch.setenv(ENV_ANALYST_ID, "analyst-prime")

        config = Config.from_env()

        # Values.
        assert config.encryption_key == key
        assert config.db_path == db_target
        assert config.upstream_base_url == "https://example.com"
        assert config.proxy_host == "0.0.0.0"
        assert config.proxy_port == 9090
        assert config.log_level == "DEBUG"
        assert config.default_analyst_id == "analyst-prime"

        # Types.
        assert isinstance(config.encryption_key, bytes)
        assert isinstance(config.db_path, Path)
        assert isinstance(config.upstream_base_url, str)
        assert isinstance(config.proxy_host, str)
        assert isinstance(config.proxy_port, int)
        assert isinstance(config.log_level, str)
        assert isinstance(config.default_analyst_id, str)

    def test_config_is_frozen(self, monkeypatch, valid_key_str):
        monkeypatch.setenv(ENV_ENCRYPTION_KEY, valid_key_str)
        config = Config.from_env()
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.proxy_port = 9999  # type: ignore[misc]


class TestGenerateLoadSeparation:
    """The loader path never calls Fernet.generate_key. Generation and
    loading are strictly separate operations."""

    def test_loader_does_not_call_generate_key(
        self, monkeypatch, valid_key_str
    ):
        # valid_key_str fixture already produced a key before this body
        # runs, so the sentinel installed below cannot break setup.
        def _sentinel(*args, **kwargs):
            pytest.fail("loader must not call Fernet.generate_key on any path")

        monkeypatch.setattr(
            "cryptography.fernet.Fernet.generate_key", _sentinel
        )

        # Path 1: missing key. ConfigError, no generate call.
        with pytest.raises(ConfigError):
            Config.from_env()

        # Path 2: malformed key. ConfigError, no generate call.
        monkeypatch.setenv(ENV_ENCRYPTION_KEY, "not_a_real_key")
        with pytest.raises(ConfigError):
            Config.from_env()

        # Path 3: valid key. Loads cleanly via Fernet() constructor,
        # which does NOT call generate_key.
        monkeypatch.setenv(ENV_ENCRYPTION_KEY, valid_key_str)
        Config.from_env()

    def test_generate_encryption_key_does_call_generate_key(
        self, monkeypatch
    ):
        # Positive control: confirms the sentinel approach above would
        # actually catch a misuse. generate_encryption_key MUST call
        # Fernet.generate_key.
        called = {"count": 0}
        original = Fernet.generate_key

        def _counting():
            called["count"] += 1
            return original()

        monkeypatch.setattr(
            "cryptography.fernet.Fernet.generate_key", _counting
        )
        key = generate_encryption_key()
        assert called["count"] == 1
        # And it returns a Fernet-valid key.
        Fernet(key)


class TestConfigErrorHierarchy:
    """ConfigError is its own root, not part of the audit-store
    exception tree."""

    def test_not_subclass_of_audit_store_error(self):
        assert not issubclass(ConfigError, AuditStoreError)

    def test_is_subclass_of_exception(self):
        assert issubclass(ConfigError, Exception)

    def test_not_caught_by_audit_store_error_handler(self):
        try:
            raise ConfigError("setup problem")
        except AuditStoreError:
            pytest.fail("ConfigError was caught by AuditStoreError handler")
        except ConfigError:
            pass


class TestStructuralGuards:
    """Module-level invariants that protect security-critical
    properties from quiet regression."""

    def test_config_module_does_not_import_logging(self):
        # Guards against a future commit adding a log line that later
        # grows to include field values (especially the encryption
        # key). The module is meant to do validation only, with no
        # logging surface. Catches both `import logging` and the
        # `logger = logging.getLogger(...)` pattern.
        assert not hasattr(llmdr.config, "logging"), (
            "llmdr.config must not import logging"
        )
        assert not hasattr(llmdr.config, "logger"), (
            "llmdr.config must not define a logger"
        )
