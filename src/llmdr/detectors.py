"""PII and credential detection over audit record content.

Two detectors, both observational (v0.1 never blocks or modifies):

* **detect-secrets** (required dependency) finds credentials: API keys,
  private keys, high entropy strings, keyword-adjacent secrets.
* **Microsoft Presidio** (optional dependency, ``pip install
  llmdr[detection]``) finds PII: names, emails, phone numbers, and the
  rest of its recognizer set. When Presidio or its spaCy model is not
  installed, PII detection is skipped and recorded as skipped; llmdr
  stays fully functional without it.

The public surface is :func:`run_detection`, which returns a
:class:`DetectionResult`, and :meth:`DetectionResult.to_metadata`, which
shapes the outcome for storage in the encrypted metadata of an audit
record.

Contracts (do not soften without an explicit decision):

* :func:`run_detection` never raises. Detection is advisory; an audit
  write must never be blocked by a detector crash. Detector failures
  are captured in the result as exception type names and surface as
  ``detection_status="failed"``.
* Hits never carry the matched text. A hit records the detector, the
  hit type, and the location (line number or character span). For
  credential hits, detect-secrets' SHA-1 correlation hash of the secret
  is kept so an investigator can match identical secrets across
  records without llmdr ever storing the secret itself. Metadata is
  encrypted at rest regardless; excluding the matched text is
  defense in depth on top of that.
* Nothing in this module logs content. Log lines may carry counts,
  detector names, hit types, and exception type names only.

Entropy limit note (the reason :func:`_scan_secrets` post-filters):
    ``detect_secrets.core.scan.scan_line`` runs entropy plugins in
    eager mode and intentionally skips their entropy limit, because its
    designed use is interactive "why was this not caught" debugging.
    Raw ``scan_line`` output therefore flags nearly every word of
    ordinary prose. We re-apply each entropy plugin's own configured
    limit to its hits, restoring normal file-scan semantics while
    keeping eager mode's ability to catch unquoted secrets in prose,
    which is exactly where secrets appear in LLM conversations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from llmdr.audit import DetectionStatus

if TYPE_CHECKING:
    from presidio_analyzer import AnalyzerEngine

logger = logging.getLogger(__name__)

DETECTOR_SECRETS = "detect_secrets"
DETECTOR_PII = "presidio"

# Cap on hits persisted per record. A pathological input (e.g. a dump
# of thousands of keys) should not balloon one encrypted metadata blob;
# the full count is always recorded, and truncation is flagged loudly
# in the metadata so an investigator knows the hit list is a prefix.
MAX_HITS_RECORDED = 200

# Presidio availability is resolved once per process and cached,
# because building an AnalyzerEngine loads a spaCy model (seconds).
# Values: None = not yet attempted, otherwise the analyzer or False.
_presidio_analyzer: AnalyzerEngine | None | bool = None


@dataclass(frozen=True)
class DetectionHit:
    """One detector finding, locating but never containing the match.

    ``line`` is 1-based and set for detect-secrets hits. ``start`` and
    ``end`` are 0-based character offsets into the scanned content and
    are set for Presidio hits. ``score`` is Presidio's confidence.
    ``secret_hash`` is detect-secrets' SHA-1 of the matched secret,
    kept for cross-record correlation; the secret itself is discarded.
    """

    detector: str
    hit_type: str
    line: int | None = None
    start: int | None = None
    end: int | None = None
    score: float | None = None
    secret_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Compact dict for metadata storage; None fields are dropped."""
        out: dict[str, Any] = {"detector": self.detector, "type": self.hit_type}
        for key in ("line", "start", "end", "score", "secret_hash"):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        return out


@dataclass(frozen=True)
class DetectionResult:
    """Outcome of running all detectors over one piece of content.

    ``status`` matches the audit store's DetectionStatus vocabulary.
    Precedence, most significant first:

    * ``ran_with_hits`` when any detector found anything. Hits are the
      signal an investigator filters on, so they win even when another
      detector failed; the failure stays visible in ``errors``.
    * ``failed`` when no hits and at least one detector raised.
      Detection is incomplete and the record says so.
    * ``skipped`` when no detector ran at all.
    * ``ran_clean`` otherwise.
    """

    status: DetectionStatus
    hits: tuple[DetectionHit, ...]
    detectors_run: tuple[str, ...]
    detectors_skipped: tuple[str, ...]
    errors: tuple[str, ...]

    def to_metadata(self) -> dict[str, Any]:
        """Shape this result for the encrypted metadata of a record.

        Returns a dict with a single ``detection`` key so it merges
        into record metadata without clobbering other fields. At most
        :data:`MAX_HITS_RECORDED` hits are stored; ``hit_count`` always
        reflects the true total and ``hits_truncated`` flags a cut.
        """
        detection: dict[str, Any] = {
            "detectors_run": list(self.detectors_run),
            "detectors_skipped": list(self.detectors_skipped),
            "errors": list(self.errors),
            "hit_count": len(self.hits),
            "hits": [h.to_dict() for h in self.hits[:MAX_HITS_RECORDED]],
        }
        if len(self.hits) > MAX_HITS_RECORDED:
            detection["hits_truncated"] = True
        return {"detection": detection}


def run_detection(content: str) -> DetectionResult:
    """Run all available detectors over ``content``. Never raises.

    detect-secrets always runs (required dependency). Presidio runs
    when importable and its NLP model loads; otherwise it is recorded
    under ``detectors_skipped``. A detector that raises mid-scan is
    recorded in ``errors`` as ``"<detector>:<ExceptionTypeName>"``;
    the exception message is deliberately dropped because scanner
    errors can quote the input they choked on, and content must not
    leak into any plaintext-adjacent path.
    """
    hits: list[DetectionHit] = []
    detectors_run: list[str] = []
    detectors_skipped: list[str] = []
    errors: list[str] = []

    try:
        hits.extend(_scan_secrets(content))
        detectors_run.append(DETECTOR_SECRETS)
    except Exception as exc:
        errors.append(f"{DETECTOR_SECRETS}:{type(exc).__name__}")
        logger.warning(
            "detect-secrets scan failed: %s", type(exc).__name__
        )

    analyzer = _get_presidio_analyzer()
    if analyzer is None:
        detectors_skipped.append(DETECTOR_PII)
    else:
        try:
            hits.extend(_scan_pii(analyzer, content))
            detectors_run.append(DETECTOR_PII)
        except Exception as exc:
            errors.append(f"{DETECTOR_PII}:{type(exc).__name__}")
            logger.warning("presidio scan failed: %s", type(exc).__name__)

    if hits:
        status: DetectionStatus = "ran_with_hits"
    elif errors:
        status = "failed"
    elif not detectors_run:
        status = "skipped"
    else:
        status = "ran_clean"

    logger.debug(
        "detection status=%s hits=%d run=%s skipped=%s errors=%s",
        status,
        len(hits),
        detectors_run,
        detectors_skipped,
        errors,
    )
    return DetectionResult(
        status=status,
        hits=tuple(hits),
        detectors_run=tuple(detectors_run),
        detectors_skipped=tuple(detectors_skipped),
        errors=tuple(errors),
    )


def _scan_secrets(content: str) -> list[DetectionHit]:
    """Scan content line by line with detect-secrets.

    Content never touches disk: detect-secrets' file-oriented API is
    bypassed in favor of per-line ``scan_line`` calls, then entropy
    plugin hits are re-filtered by each plugin's own entropy limit
    (see the module docstring for why scan_line skips it).

    Duplicate findings (the same secret flagged by the same plugin on
    the same line) collapse to one hit.
    """
    # Imported here so a detect-secrets failure is attributable to this
    # detector by run_detection's per-detector exception handling.
    from detect_secrets.core.scan import get_plugins, scan_line
    from detect_secrets.plugins.high_entropy_strings import (
        HighEntropyStringsPlugin,
    )
    from detect_secrets.settings import default_settings

    hits: list[DetectionHit] = []
    seen: set[tuple[str, int, str | None]] = set()
    with default_settings():
        entropy_plugins = {
            plugin.secret_type: plugin
            for plugin in get_plugins()
            if isinstance(plugin, HighEntropyStringsPlugin)
        }
        for lineno, line in enumerate(content.splitlines(), start=1):
            for secret in scan_line(line):
                plugin = entropy_plugins.get(secret.type)
                if plugin is not None and secret.secret_value is not None:
                    entropy = plugin.calculate_shannon_entropy(
                        secret.secret_value
                    )
                    if entropy <= plugin.entropy_limit:
                        continue
                key = (secret.type, lineno, secret.secret_hash)
                if key in seen:
                    continue
                seen.add(key)
                hits.append(
                    DetectionHit(
                        detector=DETECTOR_SECRETS,
                        hit_type=secret.type,
                        line=lineno,
                        secret_hash=secret.secret_hash,
                    )
                )
    return hits


def _scan_pii(analyzer: AnalyzerEngine, content: str) -> list[DetectionHit]:
    """Run Presidio over the full content string."""
    results = analyzer.analyze(text=content, language="en")
    return [
        DetectionHit(
            detector=DETECTOR_PII,
            hit_type=result.entity_type,
            start=result.start,
            end=result.end,
            score=round(float(result.score), 4),
        )
        for result in results
    ]


def _get_presidio_analyzer() -> AnalyzerEngine | None:
    """Return a cached AnalyzerEngine, or None when unavailable.

    Unavailable covers "presidio-analyzer not installed" (it is an
    optional extra), "no English spaCy model installed", and "engine
    failed to start". Either way the outcome is the same skip, logged
    once at WARNING; retrying per record would pay the model-load cost
    on every audit write for nothing.

    The engine is built explicitly from an already-installed spaCy
    model rather than through presidio's default constructor, because
    the default silently downloads en_core_web_lg from the network
    when it is missing. llmdr must never make a surprise network call
    (see CLAUDE.md, no phone home of any kind), so a missing model is
    a skip plus an instruction, not a download.
    """
    global _presidio_analyzer
    if _presidio_analyzer is None:
        _presidio_analyzer = _build_presidio_analyzer()
        if _presidio_analyzer is not False:
            logger.info("presidio analyzer initialized")
    return _presidio_analyzer if _presidio_analyzer is not False else None


_PREFERRED_SPACY_MODEL = "en_core_web_lg"


def _build_presidio_analyzer() -> AnalyzerEngine | bool:
    """Construct the engine from installed components, or False."""
    try:
        import spacy
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider
    except ImportError:
        logger.warning(
            "presidio-analyzer not installed; PII detection skipped. "
            "Install with: pip install llmdr[detection]"
        )
        return False

    english_models = sorted(
        model
        for model in spacy.util.get_installed_models()
        if model.startswith("en")
    )
    if not english_models:
        logger.warning(
            "presidio-analyzer is installed but no English spaCy model "
            "is; PII detection skipped. Install one with: "
            "python -m spacy download %s",
            _PREFERRED_SPACY_MODEL,
        )
        return False
    model_name = (
        _PREFERRED_SPACY_MODEL
        if _PREFERRED_SPACY_MODEL in english_models
        else english_models[0]
    )

    try:
        nlp_engine = NlpEngineProvider(
            nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": model_name}],
            }
        ).create_engine()
        analyzer = AnalyzerEngine(
            nlp_engine=nlp_engine, supported_languages=["en"]
        )
    except Exception as exc:
        logger.warning(
            "presidio engine failed to start with model %s (%s); "
            "PII detection skipped.",
            model_name,
            type(exc).__name__,
        )
        return False
    logger.info("presidio using spaCy model %s", model_name)
    return analyzer
