"""Tests for the detection layer.

Organized by what each test proves:

* TestSecretsDetection: detect-secrets finds real credential shapes
  and attributes them to the right line.
* TestEntropyPostFilter: ordinary prose does not flag. This is the
  regression guard for the scan_line eager-mode entropy quirk; if it
  fails, every audit record would light up as ran_with_hits.
* TestNoContentLeak: the matched secret text never appears anywhere in
  the result or its metadata shape.
* TestStatusPrecedence: the four DetectionStatus values are assigned
  per the documented precedence, including detector failure paths.
* TestMetadataShape: to_metadata produces the stable envelope the
  report and investigators rely on, including hit truncation.
* TestPresidio: PII detection, exercised only when presidio-analyzer
  and a spaCy model are installed; otherwise skipped, mirroring the
  runtime's own graceful degradation.
* TestNeverRaises: run_detection swallows detector crashes and reports
  them as failures instead of propagating.
"""

from __future__ import annotations

import json

import pytest

import llmdr.detectors as detectors
from llmdr.detectors import (
    DETECTOR_PII,
    DETECTOR_SECRETS,
    DetectionHit,
    DetectionResult,
    run_detection,
)

AWS_KEY = "AKIAIOSFODNN7EXAMPLE"
# High entropy string that should survive the entropy limit.
HIGH_ENTROPY_TOKEN = "zK9mQ2vX7pL4wR8tY3nB6cF1dG5hJ0aS"


def _presidio_available() -> bool:
    return detectors._get_presidio_analyzer() is not None


@pytest.fixture(autouse=True)
def _reset_presidio_cache(monkeypatch: pytest.MonkeyPatch):
    """Isolate the module-level presidio cache between tests."""
    original = detectors._presidio_analyzer
    yield
    detectors._presidio_analyzer = original


# -----------------------------------------------------------------------------
# Secrets detection
# -----------------------------------------------------------------------------


class TestSecretsDetection:
    def test_aws_key_detected(self) -> None:
        result = run_detection(f"my key is {AWS_KEY} please use it")
        types = {h.hit_type for h in result.hits}
        assert "AWS Access Key" in types
        assert result.status == "ran_with_hits"
        assert DETECTOR_SECRETS in result.detectors_run

    def test_keyword_secret_detected(self) -> None:
        result = run_detection("password = 'hunter2secret'")
        assert any(h.hit_type == "Secret Keyword" for h in result.hits)
        assert result.status == "ran_with_hits"

    def test_line_number_attribution(self) -> None:
        content = f"first line is fine\nsecond has {AWS_KEY}\nthird fine"
        result = run_detection(content)
        aws_hits = [h for h in result.hits if h.hit_type == "AWS Access Key"]
        assert aws_hits
        assert all(h.line == 2 for h in aws_hits)

    def test_secret_hash_present_for_correlation(self) -> None:
        result = run_detection(f"key: {AWS_KEY}")
        aws_hits = [h for h in result.hits if h.hit_type == "AWS Access Key"]
        assert aws_hits
        assert all(isinstance(h.secret_hash, str) and len(h.secret_hash) == 40 for h in aws_hits)

    def test_same_secret_same_line_deduplicated(self) -> None:
        result = run_detection(f"{AWS_KEY} and again {AWS_KEY}")
        aws_hits = [h for h in result.hits if h.hit_type == "AWS Access Key"]
        assert len(aws_hits) == 1

    def test_same_secret_on_two_lines_counted_per_line(self) -> None:
        result = run_detection(f"{AWS_KEY}\n{AWS_KEY}")
        aws_hits = [h for h in result.hits if h.hit_type == "AWS Access Key"]
        assert sorted(h.line for h in aws_hits) == [1, 2]


# -----------------------------------------------------------------------------
# Entropy post-filter
# -----------------------------------------------------------------------------


class TestEntropyPostFilter:
    """scan_line eager mode skips the entropy limit; we re-apply it."""

    def test_plain_prose_is_clean(self) -> None:
        result = run_detection("The meeting is at 3pm tomorrow, bring notes.")
        assert DETECTOR_SECRETS in result.detectors_run
        assert not [h for h in result.hits if h.detector == DETECTOR_SECRETS]

    def test_typical_llm_request_body_is_clean(self) -> None:
        body = json.dumps(
            {
                "model": "claude-sonnet-5",
                "max_tokens": 1024,
                "messages": [
                    {
                        "role": "user",
                        "content": "Summarize our discussion from earlier.",
                    }
                ],
            }
        )
        result = run_detection(body)
        assert not [h for h in result.hits if h.detector == DETECTOR_SECRETS]

    def test_genuinely_high_entropy_token_still_flags(self) -> None:
        result = run_detection(f"here is the value {HIGH_ENTROPY_TOKEN} as requested")
        entropy_hits = [h for h in result.hits if "Entropy" in h.hit_type]
        assert entropy_hits

    def test_empty_content_is_clean(self) -> None:
        result = run_detection("")
        assert not result.hits
        assert DETECTOR_SECRETS in result.detectors_run


# -----------------------------------------------------------------------------
# No content leak
# -----------------------------------------------------------------------------


class TestNoContentLeak:
    def test_secret_value_absent_from_result_and_metadata(self) -> None:
        result = run_detection(f"aws_key = {AWS_KEY}")
        assert result.hits
        serialized = json.dumps(result.to_metadata())
        assert AWS_KEY not in serialized
        for hit in result.hits:
            for value in vars(hit).values():
                assert value != AWS_KEY

    def test_hit_dict_has_no_text_fields(self) -> None:
        hit = DetectionHit(
            detector=DETECTOR_SECRETS,
            hit_type="AWS Access Key",
            line=1,
            secret_hash="a" * 40,
        )
        assert set(hit.to_dict().keys()) <= {
            "detector",
            "type",
            "line",
            "start",
            "end",
            "score",
            "secret_hash",
        }


# -----------------------------------------------------------------------------
# Status precedence
# -----------------------------------------------------------------------------


class TestStatusPrecedence:
    def test_clean_run_without_presidio_is_ran_clean(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(detectors, "_get_presidio_analyzer", lambda: None)
        result = run_detection("nothing sensitive here at all")
        assert result.status == "ran_clean"
        assert result.detectors_skipped == (DETECTOR_PII,)

    def test_hits_win_over_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _Boom:
            def analyze(self, **kwargs: object) -> None:
                raise RuntimeError("engine exploded")

        monkeypatch.setattr(detectors, "_get_presidio_analyzer", lambda: _Boom())
        result = run_detection(f"key {AWS_KEY}")
        assert result.status == "ran_with_hits"
        assert f"{DETECTOR_PII}:RuntimeError" in result.errors

    def test_failure_without_hits_is_failed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _boom(content: str) -> list[DetectionHit]:
            raise ValueError("scanner crashed")

        monkeypatch.setattr(detectors, "_scan_secrets", _boom)
        monkeypatch.setattr(detectors, "_get_presidio_analyzer", lambda: None)
        result = run_detection("benign")
        assert result.status == "failed"
        assert result.errors == (f"{DETECTOR_SECRETS}:ValueError",)

    def test_empty_scan_with_presidio_skipped_is_ran_clean(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A skipped presidio never downgrades ran_clean to skipped;
        # "skipped" is reserved for no detector running at all, which
        # cannot happen while detect-secrets is a required dependency.
        monkeypatch.setattr(detectors, "_scan_secrets", lambda c: [])
        monkeypatch.setattr(detectors, "_get_presidio_analyzer", lambda: None)
        result = run_detection("x")
        assert result.status == "ran_clean"
        assert result.detectors_skipped == (DETECTOR_PII,)

    def test_error_message_not_in_errors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        secret_text = "the secret content that must not leak"

        def _boom(content: str) -> list[DetectionHit]:
            raise ValueError(secret_text)

        monkeypatch.setattr(detectors, "_scan_secrets", _boom)
        result = run_detection("anything")
        assert secret_text not in json.dumps(result.errors)


# -----------------------------------------------------------------------------
# Metadata shape
# -----------------------------------------------------------------------------


class TestMetadataShape:
    def test_envelope_shape(self) -> None:
        result = run_detection(f"key {AWS_KEY}")
        meta = result.to_metadata()
        assert set(meta.keys()) == {"detection"}
        detection = meta["detection"]
        assert set(detection.keys()) == {
            "detectors_run",
            "detectors_skipped",
            "errors",
            "hit_count",
            "hits",
        }
        assert detection["hit_count"] == len(result.hits)
        assert detection["hit_count"] == len(detection["hits"])

    def test_metadata_is_json_serializable(self) -> None:
        result = run_detection(f"key {AWS_KEY}\npassword='hunter2secret'")
        json.dumps(result.to_metadata())

    def test_truncation_flag_and_true_count(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(detectors, "MAX_HITS_RECORDED", 3)
        hits = tuple(
            DetectionHit(
                detector=DETECTOR_SECRETS,
                hit_type="AWS Access Key",
                line=i,
                secret_hash=f"{i:040d}",
            )
            for i in range(1, 11)
        )
        result = DetectionResult(
            status="ran_with_hits",
            hits=hits,
            detectors_run=(DETECTOR_SECRETS,),
            detectors_skipped=(),
            errors=(),
        )
        detection = result.to_metadata()["detection"]
        assert detection["hit_count"] == 10
        assert len(detection["hits"]) == 3
        assert detection["hits_truncated"] is True

    def test_no_truncation_flag_under_cap(self) -> None:
        result = run_detection(f"key {AWS_KEY}")
        assert "hits_truncated" not in result.to_metadata()["detection"]


# -----------------------------------------------------------------------------
# Presidio (skipped when unavailable)
# -----------------------------------------------------------------------------


@pytest.mark.skipif(
    not _presidio_available(),
    reason="presidio-analyzer or its spaCy model not installed",
)
class TestPresidio:
    def test_email_detected_with_span(self) -> None:
        content = "Reach me at jane.doe@example.com about the case."
        result = run_detection(content)
        email_hits = [h for h in result.hits if h.hit_type == "EMAIL_ADDRESS"]
        assert email_hits
        hit = email_hits[0]
        assert content[hit.start : hit.end] == "jane.doe@example.com"
        assert hit.detector == DETECTOR_PII
        assert hit.score is not None and 0.0 <= hit.score <= 1.0

    def test_person_detected(self) -> None:
        result = run_detection("Please forward this to Jane Doe today.")
        assert any(h.hit_type == "PERSON" for h in result.hits)
        assert DETECTOR_PII in result.detectors_run

    def test_pii_hit_carries_no_text(self) -> None:
        result = run_detection("Reach me at jane.doe@example.com.")
        serialized = json.dumps(result.to_metadata())
        assert "jane.doe@example.com" not in serialized


class TestPresidioUnavailable:
    def test_missing_presidio_is_skip_not_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(detectors, "_presidio_analyzer", False)
        result = run_detection("hello there")
        assert DETECTOR_PII in result.detectors_skipped
        assert result.status in ("ran_clean", "ran_with_hits")


# -----------------------------------------------------------------------------
# Never raises
# -----------------------------------------------------------------------------


class TestNeverRaises:
    def test_both_detectors_crashing_still_returns(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _boom(content: str) -> list[DetectionHit]:
            raise RuntimeError("boom")

        class _Boom:
            def analyze(self, **kwargs: object) -> None:
                raise OSError("model gone")

        monkeypatch.setattr(detectors, "_scan_secrets", _boom)
        monkeypatch.setattr(detectors, "_get_presidio_analyzer", lambda: _Boom())
        result = run_detection("anything")
        assert result.status == "failed"
        assert sorted(result.errors) == [
            f"{DETECTOR_SECRETS}:RuntimeError",
            f"{DETECTOR_PII}:OSError",
        ]

    def test_unusual_content_shapes(self) -> None:
        for content in ("\x00\x01\x02", "a" * 100_000, "\n" * 500, "🔑" * 50):
            result = run_detection(content)
            assert result.status in (
                "ran_clean",
                "ran_with_hits",
                "failed",
                "skipped",
            )
