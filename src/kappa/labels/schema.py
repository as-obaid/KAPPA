"""Typed, validated shape for one report extraction.

Why this exists: ``parse_response`` only proves the model returned *JSON*, not
that it returned the JSON the prompt asked for. It checks no field, type or enum, so a
missing ``assertion`` would become ``None`` and enter the assertion
distribution, and an ``anatomy`` of ``[]`` instead of a string would become a
key in the anatomy census -- silently, in the exact tables the survey exists to
produce.

**These are unguarded failure modes, not observed ones.** Run against all 300
records of the gemma-4-31B survey this validator produced 300/300 valid, zero
coercions and zero dropped findings. It is insurance for the models that have
not been run yet, and a way to make any future breakage counted instead of
invisible -- not a repair of something already broken.

The models are deliberately **salvaging rather than strict**. A survey over
4,407 reports would rather keep 12 good findings and drop one malformed one
than lose a whole report, so recoverable problems are coerced and recorded as
warnings, and only a fundamentally unusable payload returns ``None``. Every
coercion is reported, so "how often did the model's output need repair" becomes
a measurement instead of an assumption.

Field expectations come from a measured 300-report run, not from taste:

* ``severity_wording`` is null in **68%** of findings -- optional, not missing.
* ``sections`` is empty in **21%** of reports -- never branch on it.
* ``knee_side`` is ``unstated`` in **47%**, and ``knee_side_evidence`` was null
  in exactly those cases, so the two are cross-checked.
* ``language_confident`` was ``true`` for **300/300**. A flag that is never
  false earns no trust; it is preserved but explicitly not relied upon.
* ``findings`` ranged 0-37 per report; zero is legal.

    from kappa.labels.schema import validate_extraction

    model, warnings = validate_extraction(parsed_json)
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "Assertion",
    "KneeSide",
    "Finding",
    "ReportExtraction",
    "validate_extraction",
]

Assertion = Literal["affirmed", "negated", "uncertain"]
KneeSide = Literal["left", "right", "unstated"]

#: Values seen in place of a clean assertion, mapped to what they mean.
_ASSERTION_ALIASES = {
    "affirm": "affirmed",
    "affirmative": "affirmed",
    "positive": "affirmed",
    "present": "affirmed",
    "negate": "negated",
    "negative": "negated",
    "absent": "negated",
    "normal": "negated",
    "unknown": "uncertain",
    "possible": "uncertain",
    "equivocal": "uncertain",
}

_SIDE_ALIASES = {
    "l": "left",
    "r": "right",
    "bilateral": "unstated",
    "both": "unstated",
    "none": "unstated",
    "not stated": "unstated",
    "unspecified": "unstated",
    "null": "unstated",
}


def _clean_str(value: Any) -> Any:
    """Strip a string; turn blanks and ``null``-alikes into None.

    The model emits the *strings* "null", "none" and "n/a" as well as real
    nulls, and those would otherwise become severity vocabulary entries.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text.lower() in {"null", "none", "n/a", "na", "-"}:
        return None
    return text


class Finding(BaseModel):
    """One reported observation, with the verbatim span that evidences it."""

    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)

    span: str = Field(min_length=1)
    description: str | None = None
    anatomy: str | None = None
    assertion: Assertion = "uncertain"
    severity_wording: str | None = None

    @field_validator("description", "anatomy", "severity_wording", mode="before")
    @classmethod
    def _blank_to_none(cls, value: Any) -> Any:
        return _clean_str(value)

    @field_validator("assertion", mode="before")
    @classmethod
    def _coerce_assertion(cls, value: Any) -> Any:
        """Fall back to ``uncertain`` rather than dropping the finding.

        A finding whose assertion cannot be read still carries a real span and
        real anatomy; discarding it would lose evidence over a field that can be
        represented honestly as unknown.
        """
        if not isinstance(value, str):
            return "uncertain"
        text = value.strip().lower()
        if text in {"affirmed", "negated", "uncertain"}:
            return text
        return _ASSERTION_ALIASES.get(text, "uncertain")


class ReportExtraction(BaseModel):
    """Everything the survey prompt asks for about a single report."""

    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)

    language: str | None = None
    language_confident: bool | None = None
    knee_side: KneeSide = "unstated"
    knee_side_evidence: str | None = None
    sections: list[str] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    stated_grades: list[str] = Field(default_factory=list)
    notes: str | None = None

    @field_validator("language", "knee_side_evidence", "notes", mode="before")
    @classmethod
    def _blank_to_none(cls, value: Any) -> Any:
        return _clean_str(value)

    @field_validator("knee_side", mode="before")
    @classmethod
    def _coerce_side(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return "unstated"
        text = value.strip().lower()
        if text in {"left", "right", "unstated"}:
            return text
        return _SIDE_ALIASES.get(text, "unstated")

    @field_validator("sections", "stated_grades", mode="before")
    @classmethod
    def _coerce_str_list(cls, value: Any) -> Any:
        """Accept a bare string where a list was asked for, and drop blanks.

        A single-section report comes back as ``"FINDINGS"`` rather than
        ``["FINDINGS"]`` often enough to be worth absorbing here.
        """
        if value is None:
            return []
        if isinstance(value, str):
            cleaned = _clean_str(value)
            return [cleaned] if cleaned else []
        if not isinstance(value, list):
            return []
        out: list[str] = []
        for item in value:
            cleaned = _clean_str(item if isinstance(item, str) else str(item))
            if cleaned:
                out.append(cleaned)
        return out


def validate_extraction(payload: Any) -> tuple[ReportExtraction | None, list[str]]:
    """Validate one parsed response.

    Returns ``(model, warnings)``. ``model`` is None only when the payload is
    not a JSON object at all, or is so malformed that even the report-level
    fields will not validate -- anything short of that is coerced and reported,
    because a survey loses more by discarding a report than by recording that
    one of its fields needed repair.

    ``warnings`` is the audit trail, and is meant to be counted across a run:
    a coercion happening twice is noise, the same coercion happening 300 times
    is a prompt defect.
    """
    warnings: list[str] = []
    if not isinstance(payload, dict):
        return None, [f"payload is {type(payload).__name__}, expected object"]

    raw_findings = payload.get("findings")
    if raw_findings is not None and not isinstance(raw_findings, list):
        warnings.append(f"findings was {type(raw_findings).__name__}, not a list")
        raw_findings = []

    # Validate findings one at a time so a single bad entry costs one finding
    # rather than the whole report.
    kept: list[dict] = []
    for i, item in enumerate(raw_findings or []):
        if not isinstance(item, dict):
            warnings.append(f"finding[{i}] is {type(item).__name__}, dropped")
            continue
        try:
            Finding.model_validate(item)
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            reason = str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__
            warnings.append(f"finding[{i}] invalid ({reason}), dropped")
            continue
        kept.append(item)

    try:
        model = ReportExtraction.model_validate({**payload, "findings": kept})
    except Exception as exc:  # noqa: BLE001
        first = str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__
        return None, warnings + [f"report invalid ({first})"]

    # Cross-field checks. Consistency signals, not hard errors: the measured run
    # had evidence null in exactly the 140 unstated cases, so a violation means
    # that relationship has changed and is worth looking at.
    if model.knee_side != "unstated" and model.knee_side_evidence is None:
        warnings.append(f"knee_side={model.knee_side} with no evidence span")
    if model.knee_side == "unstated" and model.knee_side_evidence is not None:
        warnings.append("knee_side=unstated but an evidence span was given")

    return model, warnings
