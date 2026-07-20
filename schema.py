"""
Pydantic schema for a clinical SOAP note.

This is the structural contract every stage of the pipeline serves. The agent
must produce output that validates against these models; anything malformed
raises a ValidationError, which the agent's retry loop uses as feedback.

SOAP = Subjective, Objective, Assessment, Plan.
"""

from __future__ import annotations

import re
from pydantic import BaseModel, Field, field_validator


ICD10_PATTERN = re.compile(r"^[A-TV-Z][0-9][0-9AB](?:\.[0-9A-Z]{1,4})?$")


class Medication(BaseModel):
    """A single medication line. Name is mandatory; dose/frequency optional."""

    name: str = Field(..., min_length=1, description="Drug name")
    dose: str | None = Field(default=None, description="e.g. '500 mg'")
    frequency: str | None = Field(default=None, description="e.g. 'twice daily'")


class SOAPNote(BaseModel):
    """A structured clinical note.

    Every field except `medications` and `icd_candidates` is required, so a
    draft that drops a whole section will fail validation and trigger a retry.
    """

    subjective: str = Field(
        ...,
        min_length=10,
        description="Patient-reported symptoms, history, and concerns.",
    )
    objective: list[str] = Field(
        ...,
        min_length=1,
        description="Observable/measurable findings: vitals, exam results.",
    )
    assessment: str = Field(
        ...,
        min_length=3,
        description="Clinical impression / differential diagnosis.",
    )
    plan: list[str] = Field(
        ...,
        min_length=1,
        description="Next steps: tests, referrals, medications, follow-up.",
    )
    medications: list[Medication] = Field(default_factory=list)
    icd_candidates: list[str] = Field(
        default_factory=list,
        description="ICD-10 codes. MUST be grounded in retrieved references.",
    )

    @field_validator("icd_candidates")
    @classmethod
    def codes_well_formed(cls, v: list[str]) -> list[str]:
        """Reject anything that isn't even shaped like an ICD-10 code.

        Note: this only checks *format*. Whether a code is *grounded* in the
        retrieved reference chunks is enforced separately, at the agent level,
        because that check needs the retrieval context which the schema can't see.
        """
        for code in v:
            if not ICD10_PATTERN.match(code):
                raise ValueError(f"'{code}' is not a valid ICD-10 code format")
        return v


class ValidationOutcome(BaseModel):
    """Result of trying to validate a raw draft against SOAPNote."""

    ok: bool
    note: SOAPNote | None = None
    errors: list[str] = Field(default_factory=list)


def validate_draft(raw: dict) -> ValidationOutcome:
    """Try to coerce a raw dict into a SOAPNote, capturing structured errors.

    The error strings produced here are the actionable feedback the agent
    injects back into the prompt on a retry.
    """
    from pydantic import ValidationError

    try:
        note = SOAPNote(**raw)
        return ValidationOutcome(ok=True, note=note, errors=[])
    except ValidationError as exc:
        errors = []
        for err in exc.errors():
            loc = ".".join(str(x) for x in err["loc"])
            errors.append(f"{loc}: {err['msg']}")
        return ValidationOutcome(ok=False, note=None, errors=errors)
