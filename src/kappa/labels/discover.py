"""Reconnaissance: ask a model what is actually in the reports.

This module deliberately does **not** map anything to the 12 competition labels.
That mapping is the design decision this pass exists to inform, and baking it in
here would answer the question with an assumption made in advance.

The prompt asks for description, not judgement: which language, which sections,
which findings are mentioned, in what words, asserted or negated, with what
severity wording, and which standardised grades are cited. What those findings
mean for the 12 columns is decided later, by a human reading the aggregate.

    from kappa.labels.discover import build_prompt, parse_response, aggregate
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from typing import Any

from kappa.labels.masking import report_mask
from kappa.labels.polarity import polarity, signals

__all__ = [
    "build_prompt",
    "span_status",
    "polarity_by_language",
    "RECORDS_COLUMNS",
    "FINDINGS_COLUMNS",
    "records_table",
    "findings_table",
    "parse_response",
    "verify_spans",
    "render_example",
    "per_script_quality",
    "aggregate",
    "SURVEY_INSTRUCTIONS",
]

#: Any run of whitespace, newlines included -- see ``_match_key``.
_ANY_WS = re.compile(r"\s+")

SURVEY_INSTRUCTIONS = """\
You are surveying knee MRI radiology reports to document how they are written.
This is a descriptive linguistic survey, not a diagnosis and not a classification
task. Do not judge severity yourself and do not map anything to any predefined
list of conditions. Report only what the text says, in the text's own words.

Return a single JSON object with exactly these keys:

{
  "language": "the language name in English, e.g. Bulgarian, Turkish, Greek",
  "language_confident": true or false,
  "knee_side": "left" | "right" | "unstated",
  "knee_side_evidence": "the exact substring stating the side, or null",
  "sections": ["exact section headings present, verbatim, in order"],
  "findings": [
    {
      "span": "the exact substring from the report, copied character for character",
      "description": "a plain English description of the finding",
      "anatomy": "the anatomical structure involved, in English",
      "assertion": "affirmed" | "negated" | "uncertain",
      "severity_wording": "the exact word(s) the report uses for degree, or null"
    }
  ],
  "stated_grades": ["any standardised grade cited verbatim, e.g. 'ICRS Grade IV'"],
  "notes": "anything structurally unusual about this report, or null"
}

Rules:
- "span" must be copied verbatim from the report. Never paraphrase or translate it.
- "description" and "anatomy" are in English even when the report is not.
- List every finding, including normal ones stated as normal (assertion negated).
- "severity_wording" is the report's own word (mild, hafif, ήπιο, минимален...),
  never your assessment. Use null when the report states no degree.
- Output only the JSON object. No commentary, no markdown fence.
"""


def build_prompt(report: str) -> str:
    return f"{SURVEY_INSTRUCTIONS}\n\nREPORT:\n\"\"\"\n{report}\n\"\"\"\n"


_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


#: Every escape sequence, valid or not, matched as a WHOLE so that pairs are
#: consumed left to right. A lookahead like ``\\(?!["\\/bfnrtu])`` is the obvious
#: way to write this and it is wrong: in a legal ``\\\\`` pair it accepts the first
#: backslash, then re-examines the SECOND one as if it began a new escape, and
#: "repairs" a document that was already valid. Group 1 is a legal single-char
#: escape, group 2 a legal ``\\uXXXX``, group 3 anything else -- i.e. the illegal
#: case, including a truncated ``\\u`` with too few hex digits.
_ESCAPE = re.compile(r'\\(?:(["\\/bfnrt])|u([0-9a-fA-F]{4})|(.))', re.S)


def repair_invalid_escapes(text: str) -> tuple[str, int]:
    """Escape stray backslashes so a document with them can be parsed.

    Measured on the full extraction run: 1 of the first 2,168 reports produced a
    response that was complete, correctly closed, and still unparseable, because
    the SOURCE REPORT carries RTF encoding artifacts and the model reproduced one
    verbatim inside a span -- ``graad 2 \\'e0 3``, where ``\\'e0`` is RTF for
    ``a``-grave. ``\\'`` is not a legal JSON escape, so one artifact in the
    corpus discarded an otherwise perfect extraction. The model had even noted
    the artifacts itself in its own ``notes`` field.

    **The stray backslash is escaped, never decoded.** Turning ``\\'e0`` into
    ``a``-grave would be the friendlier-looking fix and it would silently break
    span verification, which asks whether the span occurs in the report -- and
    the report contains the literal artifact, not the character it stands for.
    Fidelity to the source beats readability here.

    Returns the repaired text and how many escapes were fixed, so a caller can
    record that a repair happened. A silent repair is how a corpus fact stops
    being visible.
    """
    fixed = 0

    def _sub(match: "re.Match[str]") -> str:
        nonlocal fixed
        if match.group(3) is None:
            return match.group(0)          # a legal escape, left exactly as it was
        fixed += 1
        return "\\\\" + match.group(3)      # escape the stray backslash, keep the char

    return _ESCAPE.sub(_sub, text), fixed


def parse_response(text: str) -> dict | None:
    """Parse the model's JSON, tolerating a markdown fence and stray backslashes.

    Returns None rather than raising: in a survey a single unparseable response
    is a datum about the model, not a reason to lose the other 299.
    """
    cleaned = _FENCE.sub("", text).strip()
    for candidate in _parse_candidates(cleaned):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _parse_candidates(cleaned: str):
    """The forms to try, cheapest and most faithful first.

    Order matters: the untouched text is tried before any repair, so a response
    that is already valid is never altered, and the brace-slice fallback comes
    last because it can silently drop trailing content.
    """
    yield cleaned
    repaired, n = repair_invalid_escapes(cleaned)
    if n:
        yield repaired
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        sliced = cleaned[start : end + 1]
        yield sliced
        repaired_slice, n = repair_invalid_escapes(sliced)
        if n:
            yield repaired_slice


def _match_key(text: str) -> str:
    """Fold a string to the form span comparison is meaningful in.

    NFC plus full whitespace collapse -- including newlines, which
    ``normalise_report`` deliberately keeps.
    """
    return _ANY_WS.sub(" ", unicodedata.normalize("NFC", text)).strip()


def span_status(span: Any, report: str) -> tuple[bool, bool]:
    """``(verbatim, tolerant)`` for a single span. Per-finding form of below.

    Exposed because aggregate counts tell a reviewer that spans failed without
    telling them *which*, which is the only part they can act on.
    """
    if not isinstance(span, str) or not span:
        return False, False
    return span in report, _match_key(span) in _match_key(report)


def verify_spans(record: dict, report: str) -> tuple[int, int, int]:
    """Count findings whose span really occurs in the report.

    Returns ``(verbatim, tolerant, total)``.

    A span that is not in the report means the model paraphrased, which
    silently destroys traceability -- so this is reported, not assumed away.
    But a raw substring test also fails on quotes that *are* faithful, in two
    ways that both concentrate in the non-Latin scripts this survey exists to
    assess:

    * ``load_reports`` normalises reports to NFC; the model's decoded output is
      under no such guarantee, so an accented Greek quote can be character-wise
      identical and codepoint-wise different.
    * ``normalise_report`` keeps newlines, so a quote spanning a line break
      comes back with the break rendered as a space.

    Scoring either as paraphrasing would understate Greek and Cyrillic fidelity
    -- the precise failure the 58 anchors cannot reveal later, since there are
    only 2-6 per non-English language. So ``tolerant`` is the headline number
    and ``verbatim`` is kept beside it: a gap between them is a normalisation
    artifact, not paraphrasing, and it stays visible instead of being folded
    away silently.
    """
    findings = record.get("findings") or []
    spans = [f.get("span") if isinstance(f, dict) else None for f in findings]
    verbatim = sum(1 for s in spans if isinstance(s, str) and s in report)
    report_key = _match_key(report)
    tolerant = sum(
        1 for s in spans if isinstance(s, str) and _match_key(s) in report_key
    )
    return verbatim, tolerant, len(findings)


def per_script_quality(records: list[dict]) -> list[dict]:
    """Per-script parse rate, span fidelity and finding density.

    This is the table that answers the question the survey exists to answer:
    can the model read this writing system? A script with a low parse rate or
    low span fidelity is one where extraction will quietly underperform, and the
    58 anchors cannot reveal it -- there are 2-6 anchors per non-English
    language, so per-language validation is not available later.
    """
    by_script: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_script[record.get("_script", "unknown")].append(record)

    rows = []
    for script, group in sorted(by_script.items()):
        parsed = [r for r in group if r.get("_parsed")]
        spans_ok = sum(r.get("_spans_ok", 0) for r in group)
        spans_exact = sum(r.get("_spans_exact", 0) for r in group)
        spans_all = sum(r.get("_spans_total", 0) for r in group)
        n_findings = sum(len(r.get("findings") or []) for r in parsed)
        rows.append(
            {
                "script": script,
                "n": len(group),
                "parsed": len(parsed),
                "parse_rate": len(parsed) / len(group) if group else 0.0,
                "spans_ok": spans_ok,
                "spans_exact": spans_exact,
                "spans_total": spans_all,
                "span_fidelity": spans_ok / spans_all if spans_all else None,
                "span_fidelity_exact": (
                    spans_exact / spans_all if spans_all else None
                ),
                "findings_per_report": n_findings / len(parsed) if parsed else 0.0,
                "languages": Counter(
                    r.get("language", "?") for r in parsed
                ).most_common(4),
            }
        )
    return rows


#: Polarity value -> the column it counts into. `mask` vs `masked` differ
#: because the table column reads as a state and the polarity as a verdict.
_POLARITY_KEY = {"positive": "positive", "negative": "negative", "mask": "masked"}



#: Column order for the report-level W&B table.
RECORDS_COLUMNS = (
    "study_uid", "script", "language", "knee_side", "knee_side_evidence",
    "parsed", "valid", "n_findings", "spans_ok", "spans_exact", "spans_total",
    "sections", "stated_grades", "notes", "warnings", "masked", "mask_reasons",
    "n_positive", "n_negative", "n_masked", "report",
)

#: Column order for the per-finding W&B table.
FINDINGS_COLUMNS = (
    "study_uid", "script", "language", "finding_idx", "anatomy", "assertion",
    "severity_wording", "description", "polarity", "span_verbatim",
    "span_tolerant", "span",
)


def records_table(records: list[dict], sources: dict[str, str]) -> list[list]:
    """Report-level rows for W&B, in ``RECORDS_COLUMNS`` order.

    This lives here rather than in the script because it had never run. Every
    logged run to date predates the two-table split, so all of them carry the
    old single table with its 4,000-char findings blob -- the one that turned 63
    of 300 rows into invalid JSON. The replacement was written, committed, and
    then never exercised, because the code sat inside a `try/except` in a script
    that only executes on a GPU node. Moving it into the package makes the row
    shape testable without wandb, without a GPU, and without a model.

    Report text is deliberately uncapped: the longest report in the corpus is
    4,743 chars, so the old 4,000-char cap would have truncated 4 of 4,407.
    """
    rows = []
    for record in records:
        uid = record["_study_uid"]
        report_text = sources.get(uid, "")
        mask = report_mask(record, report_text=report_text)
        polarities = [
            polarity(f.get("assertion"), f.get("description"))
            for f in (record.get("findings") or [])
            if isinstance(f, dict)
        ]
        rows.append([
            uid,
            record["_script"],
            record.get("language"),
            record.get("knee_side"),
            record.get("knee_side_evidence"),
            record["_parsed"],
            record.get("_valid"),
            len(record.get("findings") or []),
            record["_spans_ok"],
            record.get("_spans_exact"),
            record["_spans_total"],
            " | ".join(record.get("sections") or []),
            " | ".join(record.get("stated_grades") or []),
            record.get("notes"),
            " | ".join(record.get("_schema_warnings") or []),
            mask.masked,
            " | ".join(mask.reasons),
            sum(1 for p in polarities if p == "positive"),
            sum(1 for p in polarities if p == "negative"),
            sum(1 for p in polarities if p == "mask"),
            report_text,
        ])
    return rows


def findings_table(records: list[dict], sources: dict[str, str]) -> list[list]:
    """One row per finding, in ``FINDINGS_COLUMNS`` order.

    The per-finding row is the unit extraction quality is actually judged in.
    It carries the span with its own verdict, so a reviewer sees *which* span
    failed rather than only how many did, and it carries the derived polarity
    beside the raw `assertion` -- because `assertion` alone is
    language-dependent and that divergence is invisible in aggregate.
    """
    rows = []
    for record in records:
        uid = record["_study_uid"]
        report_text = sources.get(uid, "")
        for i, finding in enumerate(record.get("findings") or []):
            if not isinstance(finding, dict):
                continue
            span = finding.get("span")
            verbatim, tolerant = span_status(span, report_text)
            rows.append([
                uid, record["_script"], record.get("language"), i,
                finding.get("anatomy"), finding.get("assertion"),
                finding.get("severity_wording"), finding.get("description"),
                polarity(finding.get("assertion"), finding.get("description")),
                verbatim, tolerant, span,
            ])
    return rows


def polarity_by_language(records: list[dict]) -> list[dict]:
    """Per-language polarity rates, plus the assertion asymmetry that motivates them.

    The `normality_affirmed_pct` column is the diagnostic. It measures the model
    on cases whose answer is known by construction -- the description names a
    normality word and no abnormality, so the finding is negative whatever
    `assertion` says -- and a spread across languages is the failure this
    survey exists to catch. The derived polarity columns beside it should show
    that spread collapsed.
    """
    per_lang: dict[str, dict] = defaultdict(
        lambda: {
            "findings": 0, "reports": 0, "normality_statements": 0,
            "normality_affirmed": 0, "positive": 0, "negative": 0, "masked": 0,
        }
    )
    for record in records:
        lang = record.get("language") or "?"
        row = per_lang[lang]
        row["reports"] += 1
        for finding in record.get("findings") or []:
            if not isinstance(finding, dict):
                continue
            row["findings"] += 1
            assertion = finding.get("assertion")
            description = finding.get("description")
            row[_POLARITY_KEY[polarity(assertion, description)]] += 1
            cues = signals(description)
            if cues.normal is not None and cues.abnormal is None:
                row["normality_statements"] += 1
                if assertion == "affirmed":
                    row["normality_affirmed"] += 1

    out = []
    for lang, row in sorted(per_lang.items(), key=lambda kv: -kv[1]["findings"]):
        stmts = row["normality_statements"]
        out.append({
            "language": lang,
            "findings": row["findings"],
            "reports": row["reports"],
            "normality_statements": stmts,
            "normality_affirmed_pct": 100 * row["normality_affirmed"] / stmts if stmts else 0.0,
            "positive": row["positive"],
            "negative": row["negative"],
            "masked": row["masked"],
            "positive_per_report": row["positive"] / (row["reports"] or 1),
        })
    return out


def render_example(record: dict, report: str, max_chars: int = 1100) -> str:
    """Render one report beside what the model extracted from it.

    The frequency tables cannot tell you whether the model actually understood a
    Bulgarian knee MRI report. Reading two of them can, which is why worked
    examples are in the output and not just aggregates.
    """
    out = [
        f"#### `{record.get('_study_uid', '?')}` — script: {record.get('_script', '?')}",
        "",
        "<details><summary>source report</summary>",
        "",
        "```text",
        report[:max_chars] + ("\n[...truncated]" if len(report) > max_chars else ""),
        "```",
        "",
        "</details>",
        "",
    ]
    if not record.get("_parsed"):
        out += [
            "**UNPARSEABLE.** Raw model output:",
            "",
            "```",
            (record.get("_raw") or "")[:800],
            "```",
            "",
        ]
        return "\n".join(out)

    ok, total = record.get("_spans_ok", 0), record.get("_spans_total", 0)
    out += [
        f"- **language:** {record.get('language')} "
        f"(confident: {record.get('language_confident')})",
        f"- **knee side:** {record.get('knee_side')} "
        f"— evidence: {record.get('knee_side_evidence')!r}",
        f"- **sections:** {record.get('sections')}",
        f"- **stated grades:** {record.get('stated_grades')}",
        f"- **span fidelity:** {ok}/{total} verbatim",
        "",
        "| anatomy | assertion | severity wording | description | span verbatim |",
        "| --- | --- | --- | --- | --- |",
    ]
    for finding in (record.get("findings") or [])[:14]:
        if not isinstance(finding, dict):
            continue
        span = finding.get("span")
        verbatim = "yes" if isinstance(span, str) and span in report else "**NO**"
        desc = str(finding.get("description", ""))[:70].replace("|", "/")
        anat = str(finding.get("anatomy", ""))[:34].replace("|", "/")
        sev = str(finding.get("severity_wording"))[:24].replace("|", "/")
        out.append(
            f"| {anat} | {finding.get('assertion')} | {sev} | {desc} | {verbatim} |"
        )
    out.append("")
    if record.get("notes"):
        out += [f"> model note: {record['notes']}", ""]
    return "\n".join(out)


def aggregate(records: list[dict], reports: dict[str, str] | None = None) -> str:
    """Render the survey into a document a human can make a decision from.

    ``reports`` maps study uid to source text; when supplied, worked examples are
    included. Without it the output is aggregate-only, which is weaker: the
    frequency tables say what vocabulary exists, not whether the model read it
    correctly.
    """
    reports = reports or {}
    parsed = [r for r in records if r.get("_parsed")]

    out: list[str] = [
        "# Report reconnaissance",
        "",
        f"Sampled **{len(records)}** unlabelled reports; "
        f"**{len(parsed)}** returned parseable JSON.",
        "",
        "Descriptive only. Nothing here is mapped to the 12 competition labels -- "
        "that mapping is the decision this survey exists to inform.",
        "",
        "Every number is over this sample. Per-script counts are stated so it is "
        "visible which conclusions rest on 150 reports and which on 50. The 58 "
        "annotated studies are excluded, so none of this contaminates validation.",
        "",
        "## 1. Can the model read each script?",
        "",
        "The question the survey exists to answer. Low parse rate or low span "
        "fidelity in a script means extraction will quietly underperform there -- "
        "and with 2-6 anchors per non-English language, later validation cannot "
        "detect it.",
        "",
        "| script | n | parsed | span fidelity | byte-exact | findings/report "
        "| languages seen |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in per_script_quality(records):
        fidelity = (
            f"{100 * row['span_fidelity']:.0f}% ({row['spans_ok']}/{row['spans_total']})"
            if row["span_fidelity"] is not None
            else "n/a"
        )
        exact = (
            f"{100 * row['span_fidelity_exact']:.0f}%"
            if row.get("span_fidelity_exact") is not None
            else "n/a"
        )
        langs = " · ".join(f"{k} {v}" for k, v in row["languages"])
        out.append(
            f"| {row['script']} | {row['n']} | {row['parsed']} "
            f"({100 * row['parse_rate']:.0f}%) | {fidelity} | {exact} | "
            f"{row['findings_per_report']:.1f} | {langs} |"
        )
    out += [
        "",
        "**Span fidelity** is the share of findings whose quoted span occurs in the "
        "report after NFC and whitespace folding -- the comparison reports are "
        "already normalised under, so a faithful quote is not scored as a "
        "paraphrase for being composed differently or crossing a line break. "
        "**Below ~95% the model is paraphrasing, and evidence spans cannot be "
        "trusted for traceability.** *byte-exact* is the stricter raw-substring "
        "share. A gap between the columns is a Unicode or whitespace artifact, "
        "not paraphrasing. Measured on 300 reports it is **largest in Latin** "
        "(99% vs 92%) and absent in Turkish, so it tracks line-wrapping in the "
        "source rather than the writing system -- do not assume it favours the "
        "non-Latin scripts.",
        "",
    ]

    out += ["## 2. Worked examples", ""]
    if reports:
        by_script: dict[str, list[dict]] = defaultdict(list)
        for record in records:
            by_script[record.get("_script", "unknown")].append(record)
        for script, group in sorted(by_script.items()):
            out += [f"### {script}", ""]
            # One clean example and one problem case, so quality is visible from
            # both ends rather than only where it worked.
            clean = [r for r in group if r.get("_parsed")
                     and r.get("_spans_ok") == r.get("_spans_total")]
            problem = [r for r in group if not r.get("_parsed")
                       or r.get("_spans_ok") != r.get("_spans_total")]
            for record in (clean[:1] + problem[:1]) or group[:1]:
                out.append(render_example(record, reports.get(record.get("_study_uid"), "")))
    else:
        out += ["_(source reports not supplied; aggregates only)_", ""]

    out += ["## 3. Section headings, by frequency", ""]
    heads = Counter(h for r in parsed for h in (r.get("sections") or []) if isinstance(h, str))
    out += [f"- `{h}` — {n}" for h, n in heads.most_common(40)] + [""]

    out += ["## 4. Anatomy mentioned, by frequency", ""]
    anat = Counter(
        (f.get("anatomy") or "?").strip().lower()
        for r in parsed for f in (r.get("findings") or []) if isinstance(f, dict)
    )
    out += [f"- {a} — {n}" for a, n in anat.most_common(60)] + [""]

    out += [
        "## 5. Severity wording, per language",
        "",
        "The vocabulary an extraction rubric would have to cover. Authored ladders "
        "are guesses; this is what the corpus actually says.",
        "",
    ]
    sev: dict[str, Counter] = defaultdict(Counter)
    for record in parsed:
        lang = record.get("language", "?")
        for finding in record.get("findings") or []:
            if isinstance(finding, dict) and finding.get("severity_wording"):
                sev[lang][str(finding["severity_wording"]).strip().lower()] += 1
    for lang, counter in sorted(sev.items(), key=lambda kv: -sum(kv[1].values())):
        out += [f"### {lang} ({sum(counter.values())} findings with a degree word)", ""]
        out += [f"- {w} — {n}" for w, n in counter.most_common(30)] + [""]

    out += ["## 6. Assertion distribution", ""]
    assertion = Counter(
        (f.get("assertion") or "?")
        for r in parsed for f in (r.get("findings") or []) if isinstance(f, dict)
    )
    total_f = sum(assertion.values()) or 1
    out += [f"- {k}: {v} ({100 * v / total_f:.0f}%)" for k, v in assertion.most_common()]
    out += [
        "",
        "A high `negated` share is expected and important: reports state normals "
        "explicitly. It is also the distinction that decides whether an unmentioned "
        "structure should be a 0 or masked.",
        "",
    ]

    out += [
        "## 6b. Polarity, per language — the assertion asymmetry check",
        "",
        "`assertion` alone means two different things and which one tracks the "
        "report's language: a pure normality statement is labelled `affirmed` 93% "
        "of the time in English and 30% in Greek. This table is the standing check "
        "on that, because per-language validation is unavailable later -- the 58 "
        "labelled anchors give 2-6 studies per non-English language and none in "
        "French.",
        "",
        "`norm-stmt aff%` is the share of pure normality statements (description "
        "names a normality word and no abnormality) that the model called "
        "`affirmed`. **A spread across languages is the bug**; the derived "
        "polarity columns should show none. `pos/report` is the quantity that "
        "becomes a label and should be roughly flat across languages.",
        "",
        "| language | findings | reports | norm-stmt n | norm-stmt aff% | "
        "positive | negative | masked | pos/report |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in polarity_by_language(parsed):
        out.append(
            f"| {row['language']} | {row['findings']} | {row['reports']} | "
            f"{row['normality_statements']} | {row['normality_affirmed_pct']:.0f}% | "
            f"{row['positive']} | {row['negative']} | {row['masked']} | "
            f"{row['positive_per_report']:.1f} |"
        )
    out += [
        "",
        "Counts are indicative, not reproducible: vLLM output varies with batch "
        "composition.",
        "",
    ]

    masks = [report_mask(r) for r in parsed]
    n_masked = sum(1 for m in masks if m.masked)
    reason_counts = Counter(reason for m in masks for reason in m.reasons)
    out += [
        "## 6c. Reports masked, by reason",
        "",
        f"- masked: {n_masked} of {len(parsed)} "
        f"({100 * n_masked / (len(parsed) or 1):.1f}%)",
    ]
    out += [f"- {reason}: {n}" for reason, n in reason_counts.most_common()]
    out += [
        "",
        "A masked report is one whose 12-label row is unknowable, not one that is "
        "all-negative. `multiple_studies` is a flag rather than grounds to drop the "
        "row on its own.",
        "",
    ]

    out += ["## 7. Knee side", ""]
    sides = Counter(r.get("knee_side", "?") for r in parsed)
    out += [f"- {k}: {v}" for k, v in sides.most_common()]
    out += [
        "",
        "Recoverable side matters beyond this survey: the DICOM `Laterality` tag is "
        "present on 47.3% of series and `ImageLaterality` on 0%, so the report may be "
        "the better source for train studies (open question D3).",
        "",
    ]

    out += ["## 8. Standardised grades cited", ""]
    grades = Counter(str(g).strip() for r in parsed for g in (r.get("stated_grades") or []) if g)
    out += ([f"- `{g}` — {n}" for g, n in grades.most_common(40)] or ["- none found"])
    out += [
        "",
        "Corpus-wide, `grade|grado|derece|βαθµ` + numeral appears in 1,028 reports "
        "(23.3%), ICRS in 88, Insall-Salvati 62, Outerbridge 14. Where present these "
        "are the most reliable severity signal available.",
        "",
    ]

    notes = [r["notes"] for r in parsed if r.get("notes")]
    if notes:
        out += ["## 9. Structural oddities flagged by the model", ""]
        out += [f"- {n}" for n in notes[:40]] + [""]

    return "\n".join(out)
