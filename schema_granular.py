"""
Granular clinical schema — the state object the coverage checker reads.

WHY THIS EXISTS
---------------
The original SOAPNote had four coarse fields:

    subjective: str
    objective:  list[str]      <- a blob
    assessment: str
    plan:       list[str]      <- a blob

You cannot detect a gap in a blob. If the model captured temperature but missed
the pharynx exam, `objective` is still a non-empty list, so the note "passes".
The information is silently absent and nobody knows.

That is the dangerous failure: a note that LOOKS complete but isn't. The clinician
assumes it was checked. It wasn't.

THE FIX (this is the design point)
----------------------------------
Make every clinically-required item its OWN field, so `None` is meaningful:

    temperature:        str | None = None    <- None means NOT CAPTURED
    pharynx:            str | None = None
    tonsillar_exudate:  str | None = None

Now a gap is detectable with a single line of plain code — `if field is None` —
with no LLM judgment involved at all.

**The schema knows what a complete note contains. The model does not have to.**

This is deliberate. Asking an LLM "what did you miss?" is asking it to know what
it doesn't know, which is exactly the thing it is worst at — it will confabulate
a plausible-sounding answer. The schema, by contrast, is a fixed checklist. It
cannot hallucinate a field.

    LLM's job:     fill in what was actually said
    Schema's job:  know what SHOULD be there
    Checker's job: compare the two   (deterministic, auditable, boring)

Boring is the point. The part that has to be right is the part with no judgment
in it.

WHAT COUNTS AS "REQUIRED"
-------------------------
Not every field applies to every encounter. A knee injury has no pharynx exam.
So fields carry a `required` flag in their metadata, and the coverage checker only
flags a missing field if it was required *for this encounter type*.

v1 keeps this simple: a small core set is always required, the rest are optional.
Deciding per-specialty requirement sets is a clinical question, not an engineering
one, and I am not qualified to answer it.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field, field_validator


ICD10_PATTERN = re.compile(r"^[A-TV-Z][0-9][0-9AB](?:\.[0-9A-Z]{1,4})?$")


# --------------------------------------------------------------------- helpers

def clinical_field(description: str, required: bool = False) -> Any:
    """A nullable clinical field that declares whether it is required.

    `None` is the signal for "not captured in the conversation" — NOT for
    "captured as absent". Those are different, and conflating them is a real
    clinical error: "no tonsillar exudate" is a finding; "we never looked" is a
    gap. The model is instructed to write "not discussed" nowhere — it simply
    leaves the field null, and the checker reports it.
    """
    return Field(
        default=None,
        description=description,
        json_schema_extra={"required_for_coverage": required},
    )


# ------------------------------------------------------------------ sub-models

class Subjective(BaseModel):
    """What the patient reports. Their words, our structure."""

    chief_complaint: str | None = clinical_field(
        "The main reason for the visit, in one line.", required=True)
    history_present_illness: str | None = clinical_field(
        "Narrative of how the current problem developed.", required=True)
    duration: str | None = clinical_field(
        "How long symptoms have been present, e.g. 'three days'.", required=True)
    associated_symptoms: str | None = clinical_field(
        "Other symptoms the patient reports alongside the main complaint.")
    denies: str | None = clinical_field(
        "Symptoms the patient explicitly denied when asked.")
    relevant_history: str | None = clinical_field(
        "Past medical history, medications, allergies mentioned in this visit.")


class Objective(BaseModel):
    """What was measured or observed. Verified, not reported."""

    temperature: str | None = clinical_field("e.g. '100.2 F'", required=True)
    blood_pressure: str | None = clinical_field("e.g. '128/82'")
    heart_rate: str | None = clinical_field("e.g. '78 bpm'")
    respiratory_rate: str | None = clinical_field("e.g. '16'")
    oxygen_saturation: str | None = clinical_field("e.g. '98% on room air'")
    general_appearance: str | None = clinical_field(
        "e.g. 'alert, no acute distress'")
    exam_findings: str | None = clinical_field(
        "Physical exam findings stated by the clinician.", required=True)


class Assessment(BaseModel):
    """The clinician's judgment."""

    primary_diagnosis: str | None = clinical_field(
        "The main clinical impression.", required=True)
    reasoning: str | None = clinical_field(
        "Why this diagnosis, given the findings.")
    differential: str | None = clinical_field(
        "Other diagnoses considered, if discussed.")


class Plan(BaseModel):
    """What happens next."""

    treatment: str | None = clinical_field(
        "Medications, procedures, therapies ordered.", required=True)
    patient_instructions: str | None = clinical_field(
        "What the patient was told to do.", required=True)
    follow_up: str | None = clinical_field(
        "When to return, or under what conditions.", required=True)
    referrals: str | None = clinical_field("Specialists referred to, if any.")
    tests_ordered: str | None = clinical_field("Labs or imaging ordered, if any.")


class Medication(BaseModel):
    name: str = Field(..., min_length=1)
    dose: str | None = None
    frequency: str | None = None


# ------------------------------------------------------------------- the note

class GranularSOAPNote(BaseModel):
    """A SOAP note as a *state object*, not a blob of prose.

    Every clinical item is its own nullable field, so the coverage checker can
    ask a boring, deterministic question of each one: is this null?
    """

    subjective: Subjective = Field(default_factory=Subjective)
    objective: Objective = Field(default_factory=Objective)
    assessment: Assessment = Field(default_factory=Assessment)
    plan: Plan = Field(default_factory=Plan)

    medications: list[Medication] = Field(default_factory=list)
    icd_candidates: list[str] = Field(default_factory=list)

    @field_validator("icd_candidates")
    @classmethod
    def codes_well_formed(cls, v: list[str]) -> list[str]:
        for code in v:
            if not ICD10_PATTERN.match(code):
                raise ValueError(f"'{code}' is not a valid ICD-10 code format")
        return v


# ------------------------------------------------------- the required-field map

def required_fields() -> list[tuple[str, str]]:
    """Every (section, field) pair marked required, read from the schema itself.

    Note what this does NOT do: it does not hardcode a list. The requirement lives
    in the field definition, so the checklist and the schema cannot drift apart.
    Add a required field to the schema and the checker picks it up for free.
    """
    out: list[tuple[str, str]] = []
    sections = {
        "subjective": Subjective,
        "objective": Objective,
        "assessment": Assessment,
        "plan": Plan,
    }
    for section_name, model in sections.items():
        for field_name, info in model.model_fields.items():
            extra = info.json_schema_extra or {}
            if extra.get("required_for_coverage"):
                out.append((section_name, field_name))
    return out


def field_description(section: str, field: str) -> str:
    """Human-readable description of a field, for phrasing questions about it."""
    models = {
        "subjective": Subjective, "objective": Objective,
        "assessment": Assessment, "plan": Plan,
    }
    info = models[section].model_fields.get(field)
    return (info.description or field) if info else field
