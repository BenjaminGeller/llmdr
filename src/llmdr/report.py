"""HTML investigator report generation.

Renders a self-contained forensic report from the audit store: chain
verification badge, summary statistics, detection overview, and a
record timeline. Built for the investigator reconstructing what
happened, not for a live dashboard.

Access logging: every report reads the store exclusively through
:meth:`AuditStore.verify_chain` and :meth:`AuditStore.get_records`, so
one report generation produces two access_log entries (the
verification and the read), both carrying the analyst identity and
justification. There is no side channel into the database from this
module.

Content policy: by default the report contains NO decrypted message
content, only metadata, statistics, and detection hit locations. An
investigator who needs content in the report opts in explicitly with
``include_content=True``, which the CLI surfaces as a flag with a loud
warning: the resulting HTML file holds plaintext outside the encrypted
store and must be handled like the evidence it is. This mirrors the
``show --full`` contract.

Rendering safety: the Jinja2 environment autoescapes everything.
Record content, conversation ids, model names, and hit types are all
attacker-influenced text (an adversary can put HTML in a prompt), so
nothing reaches the page unescaped. Presidio hit spans are highlighted
by splitting content into segments in Python and wrapping hit segments
in ``<mark>`` at the template layer, never by injecting markup into
the content string itself.

The report is fully self-contained: inline CSS, no external scripts,
fonts, or images. A forensic artifact must render identically offline
and must not phone home when opened.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from jinja2 import Environment

from llmdr.audit import AuditRecord, AuditStore, VerifyResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReportOutput:
    """Result of one report generation.

    ``html`` is the complete document. ``verify`` is the chain
    verification result the report embeds, exposed so callers (the
    CLI) can echo chain state without re-verifying or scraping the
    HTML. ``record_count`` is the number of records the report covers.
    """

    html: str
    verify: VerifyResult
    record_count: int


@dataclass(frozen=True)
class _ContentSegment:
    """One run of record content, marked when inside a PII hit span."""

    text: str
    highlighted: bool


@dataclass(frozen=True)
class _RecordView:
    """Template-facing view of one audit record."""

    id: int
    ts: str
    source: str
    conversation_id: str
    role: str
    model: str
    detection_status: str
    hits: list[dict[str, Any]]
    hits_truncated: bool
    detection_errors: list[str]
    segments: list[_ContentSegment] | None


def generate_report(
    store: AuditStore,
    *,
    analyst_id: str,
    justification: str,
    case_ref: str | None = None,
    last: int | None = None,
    conversation_id: str | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    include_content: bool = False,
) -> ReportOutput:
    """Build the investigator report.

    Args:
        store: An open AuditStore. The caller owns its lifecycle.
        analyst_id: Required identity of the requesting analyst,
            recorded on both access_log entries this call produces.
        justification: Required business justification, recorded
            likewise.
        case_ref: Optional case or ticket reference for the read entry.
        last: Limit to the most recent N records, matching get_records.
        conversation_id: Restrict to one conversation.
        since: Inclusive timezone-aware lower bound on record ts.
        until: Inclusive timezone-aware upper bound on record ts.
        include_content: When True, decrypted content is embedded in
            the report with PII hit spans highlighted. Off by default;
            see the module docstring for the handling implications.

    Returns:
        A :class:`ReportOutput` carrying the complete self-contained
        HTML document, the chain verification result, and the record
        count.

    Raises:
        ValueError: Propagated from the store for empty analyst_id or
            justification, or naive datetimes.
        DecryptionError: Propagated from the store when a record cannot
            be decrypted.
    """
    verify_result = store.verify_chain(
        analyst_id=analyst_id,
        justification=f"report generation: {justification}",
    )
    records = store.get_records(
        last=last,
        conversation_id=conversation_id,
        since=since,
        until=until,
        analyst_id=analyst_id,
        justification=f"report generation: {justification}",
        case_ref=case_ref,
    )

    views = [_build_record_view(r, include_content) for r in records]
    stats = _build_stats(records)
    filters = {
        "last": last,
        "conversation_id": conversation_id,
        "since": since.isoformat() if since is not None else None,
        "until": until.isoformat() if until is not None else None,
    }
    active_filters = {k: v for k, v in filters.items() if v is not None}

    html = _render(
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        analyst_id=analyst_id,
        justification=justification,
        case_ref=case_ref,
        verify=verify_result,
        chain_intact=verify_result.ok,
        records=views,
        stats=stats,
        active_filters=active_filters,
        include_content=include_content,
    )
    logger.info(
        "generated report: %d records, chain_intact=%s, " "include_content=%s, analyst=%s",
        len(records),
        verify_result.ok,
        include_content,
        analyst_id,
    )
    return ReportOutput(html=html, verify=verify_result, record_count=len(records))


def _build_record_view(record: AuditRecord, include_content: bool) -> _RecordView:
    detection = record.metadata.get("detection", {})
    hits = list(detection.get("hits", []))
    segments = _segment_content(record.content, hits) if include_content else None
    return _RecordView(
        id=record.id,
        ts=record.ts,
        source=record.source,
        conversation_id=record.conversation_id,
        role=record.role,
        model=record.model,
        detection_status=record.detection_status,
        hits=hits,
        hits_truncated=bool(detection.get("hits_truncated", False)),
        detection_errors=list(detection.get("errors", [])),
        segments=segments,
    )


def _segment_content(content: str, hits: list[dict[str, Any]]) -> list[_ContentSegment]:
    """Split content into plain and highlighted runs from hit spans.

    Only hits carrying character offsets (Presidio) produce highlight
    spans; detect-secrets hits are line-scoped and surface in the hit
    list next to the record instead. Overlapping or adjacent spans are
    merged, and malformed spans (out of range, inverted) are dropped
    rather than allowed to corrupt the segmentation.
    """
    spans: list[tuple[int, int]] = []
    for hit in hits:
        start, end = hit.get("start"), hit.get("end")
        if isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(content):
            spans.append((start, end))
    if not spans:
        return [_ContentSegment(text=content, highlighted=False)]

    spans.sort()
    merged: list[tuple[int, int]] = [spans[0]]
    for start, end in spans[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    segments: list[_ContentSegment] = []
    cursor = 0
    for start, end in merged:
        if start > cursor:
            segments.append(_ContentSegment(text=content[cursor:start], highlighted=False))
        segments.append(_ContentSegment(text=content[start:end], highlighted=True))
        cursor = end
    if cursor < len(content):
        segments.append(_ContentSegment(text=content[cursor:], highlighted=False))
    return segments


def _build_stats(records: list[AuditRecord]) -> dict[str, Any]:
    """Aggregate summary statistics over the selected records."""
    by_source: dict[str, int] = {}
    by_role: dict[str, int] = {}
    by_model: dict[str, int] = {}
    by_detection_status: dict[str, int] = {}
    hit_type_counts: dict[str, int] = {}
    records_with_hits = 0
    total_hits = 0

    for record in records:
        by_source[record.source] = by_source.get(record.source, 0) + 1
        by_role[record.role] = by_role.get(record.role, 0) + 1
        by_model[record.model] = by_model.get(record.model, 0) + 1
        by_detection_status[record.detection_status] = (
            by_detection_status.get(record.detection_status, 0) + 1
        )
        detection = record.metadata.get("detection", {})
        hits = detection.get("hits", [])
        # hit_count is the true total even when the stored hit list was
        # truncated at write time; prefer it for the aggregate numbers.
        record_hit_count = detection.get("hit_count", len(hits))
        if record_hit_count:
            records_with_hits += 1
            total_hits += record_hit_count
        for hit in hits:
            label = f"{hit.get('detector', '?')}: {hit.get('type', '?')}"
            hit_type_counts[label] = hit_type_counts.get(label, 0) + 1

    return {
        "total": len(records),
        "first_ts": records[0].ts if records else None,
        "last_ts": records[-1].ts if records else None,
        "conversations": len({r.conversation_id for r in records}),
        "by_source": dict(sorted(by_source.items())),
        "by_role": dict(sorted(by_role.items())),
        "by_model": dict(sorted(by_model.items())),
        "by_detection_status": dict(sorted(by_detection_status.items())),
        "hit_type_counts": dict(sorted(hit_type_counts.items(), key=lambda kv: -kv[1])),
        "records_with_hits": records_with_hits,
        "total_hits": total_hits,
    }


def _render(**context: Any) -> str:
    """Render the embedded template with autoescaping always on.

    The template lives in this module as a string rather than a
    package data file, keeping the whole report pipeline reviewable in
    one place per the project file layout. autoescape=True is set
    explicitly because Jinja2's default for string templates is off,
    and record content is attacker-influenced text.
    """
    env = Environment(autoescape=True)
    return env.from_string(_TEMPLATE_SOURCE).render(**context)


_TEMPLATE_SOURCE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>llmdr investigator report</title>
<style>
  :root {
    --ok: #1a7f37; --bad: #b42318; --warn: #b54708;
    --ink: #1f2328; --muted: #57606a; --line: #d0d7de;
    --bg: #ffffff; --panel: #f6f8fa; --mark: #fff1a8;
  }
  body {
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    color: var(--ink); background: var(--bg);
    margin: 2rem auto; max-width: 72rem; padding: 0 1rem;
    line-height: 1.45;
  }
  h1 { font-size: 1.5rem; margin-bottom: 0.25rem; }
  h2 { font-size: 1.1rem; margin-top: 2rem; border-bottom: 1px solid var(--line); padding-bottom: 0.25rem; }
  .muted { color: var(--muted); font-size: 0.85rem; }
  .badge {
    display: inline-block; padding: 0.35rem 0.9rem; border-radius: 6px;
    font-weight: 700; font-size: 1rem; color: #fff; margin: 0.5rem 0;
  }
  .badge.ok { background: var(--ok); }
  .badge.bad { background: var(--bad); }
  .pill {
    display: inline-block; padding: 0.05rem 0.5rem; border-radius: 999px;
    font-size: 0.75rem; border: 1px solid var(--line); background: var(--panel);
  }
  .pill.hit { border-color: var(--warn); color: var(--warn); font-weight: 600; }
  .pill.failed { border-color: var(--bad); color: var(--bad); font-weight: 600; }
  table { border-collapse: collapse; width: 100%; font-size: 0.85rem; }
  th, td { border: 1px solid var(--line); padding: 0.35rem 0.55rem; text-align: left; vertical-align: top; }
  th { background: var(--panel); }
  .num { text-align: right; }
  .statgrid { display: flex; flex-wrap: wrap; gap: 1rem; margin: 1rem 0; }
  .stat { background: var(--panel); border: 1px solid var(--line); border-radius: 6px; padding: 0.6rem 1rem; min-width: 8rem; }
  .stat .v { font-size: 1.4rem; font-weight: 700; }
  .stat .k { font-size: 0.75rem; color: var(--muted); }
  .content {
    white-space: pre-wrap; word-break: break-word;
    font-family: ui-monospace, "SF Mono", Consolas, monospace;
    font-size: 0.8rem; background: var(--panel);
    border: 1px solid var(--line); border-radius: 6px;
    padding: 0.5rem; margin: 0.25rem 0 0.75rem;
  }
  mark { background: var(--mark); outline: 1px solid var(--warn); }
  .notice {
    border-left: 4px solid var(--warn); background: var(--panel);
    padding: 0.5rem 0.9rem; margin: 1rem 0; font-size: 0.85rem;
  }
</style>
</head>
<body>

<h1>llmdr investigator report</h1>
<p class="muted">
  Generated {{ generated_at }} by analyst <strong>{{ analyst_id }}</strong>.
  Justification: {{ justification }}.
  {% if case_ref %}Case reference: {{ case_ref }}.{% endif %}
  This generation wrote access log entries for the chain verification
  and the record read it performed.
</p>

{% if chain_intact %}
<div class="badge ok">CHAIN INTACT</div>
{% else %}
<div class="badge bad">CHAIN BROKEN</div>
{% endif %}
<p class="muted">
  {% if chain_intact %}
    Both hash chains verified: every record's hash matches its stored
    contents and links to its predecessor.
  {% else %}
    {{ verify.message }}.
    Records at and after the break point must be treated as suspect.
  {% endif %}
</p>

{% if active_filters %}
<p class="muted">Filters applied:
  {% for k, v in active_filters.items() %}{{ k }}={{ v }}{% if not loop.last %}, {% endif %}{% endfor %}
</p>
{% endif %}

<h2>Summary</h2>
<div class="statgrid">
  <div class="stat"><div class="v">{{ stats.total }}</div><div class="k">records</div></div>
  <div class="stat"><div class="v">{{ stats.conversations }}</div><div class="k">conversations</div></div>
  <div class="stat"><div class="v">{{ stats.records_with_hits }}</div><div class="k">records with hits</div></div>
  <div class="stat"><div class="v">{{ stats.total_hits }}</div><div class="k">total detector hits</div></div>
</div>
{% if stats.total %}
<p class="muted">Time range: {{ stats.first_ts }} to {{ stats.last_ts }}</p>
<table>
  <tr><th>Dimension</th><th>Breakdown</th></tr>
  <tr><td>Source</td><td>
    {% for k, v in stats.by_source.items() %}{{ k }}: {{ v }}{% if not loop.last %}, {% endif %}{% endfor %}
  </td></tr>
  <tr><td>Role</td><td>
    {% for k, v in stats.by_role.items() %}{{ k }}: {{ v }}{% if not loop.last %}, {% endif %}{% endfor %}
  </td></tr>
  <tr><td>Model</td><td>
    {% for k, v in stats.by_model.items() %}{{ k }}: {{ v }}{% if not loop.last %}, {% endif %}{% endfor %}
  </td></tr>
  <tr><td>Detection status</td><td>
    {% for k, v in stats.by_detection_status.items() %}{{ k }}: {{ v }}{% if not loop.last %}, {% endif %}{% endfor %}
  </td></tr>
</table>
{% endif %}

{% if stats.hit_type_counts %}
<h2>Detection hits by type</h2>
<table>
  <tr><th>Detector and type</th><th class="num">Count</th></tr>
  {% for label, count in stats.hit_type_counts.items() %}
  <tr><td>{{ label }}</td><td class="num">{{ count }}</td></tr>
  {% endfor %}
</table>
{% endif %}

<h2>Timeline</h2>
{% if not include_content %}
<p class="muted">
  Message content is omitted from this report by default and remains
  only in the encrypted audit store. Regenerate with content included
  to embed decrypted content.
</p>
{% else %}
<div class="notice">
  This report embeds decrypted message content. The HTML file now
  holds plaintext outside the encrypted audit store; handle and store
  it as evidence.
</div>
{% endif %}
{% if records %}
<table>
  <tr>
    <th>id</th><th>ts</th><th>source</th><th>conversation</th>
    <th>role</th><th>model</th><th>detection</th><th>hits</th>
  </tr>
  {% for r in records %}
  <tr>
    <td class="num">{{ r.id }}</td>
    <td>{{ r.ts }}</td>
    <td>{{ r.source }}</td>
    <td>{{ r.conversation_id }}</td>
    <td>{{ r.role }}</td>
    <td>{{ r.model }}</td>
    <td>
      {% if r.detection_status == 'ran_with_hits' %}
        <span class="pill hit">{{ r.detection_status }}</span>
      {% elif r.detection_status == 'failed' %}
        <span class="pill failed">{{ r.detection_status }}</span>
      {% else %}
        <span class="pill">{{ r.detection_status }}</span>
      {% endif %}
    </td>
    <td>
      {% for h in r.hits %}
        <span class="pill hit">{{ h.type }}{% if h.line %} L{{ h.line }}{% endif %}</span>
      {% endfor %}
      {% if r.hits_truncated %}<span class="pill">hit list truncated</span>{% endif %}
      {% for e in r.detection_errors %}<span class="pill failed">{{ e }}</span>{% endfor %}
    </td>
  </tr>
  {% if include_content %}
  <tr>
    <td colspan="8">
      <div class="content">{% for seg in r.segments %}{% if seg.highlighted %}<mark>{{ seg.text }}</mark>{% else %}{{ seg.text }}{% endif %}{% endfor %}</div>
    </td>
  </tr>
  {% endif %}
  {% endfor %}
</table>
{% else %}
<p class="muted">No records matched the selection.</p>
{% endif %}

<p class="muted">
  Report generated by llmdr. Verify chain integrity independently at
  any time with: llmdr verify
</p>

</body>
</html>
"""
