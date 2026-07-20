"""
Coverage detection and clarification — making the system know what it doesn't know.

THE PROBLEM THIS SOLVES
-----------------------
The pipeline captures ~54% of the reference content on an external benchmark. The
tempting fix is to push the prompt harder: "be more thorough", "capture everything".

That fix is wrong, and it is wrong in a dangerous direction. Pressuring a model to
be more complete on information that ISN'T IN THE TRANSCRIPT pushes it to invent.
In clinical documentation, a fabricated finding is a far worse failure than a
missing one — a missing finding is visible if you look for it; a fabricated one
looks exactly like a real one.

So the goal is not a more complete first pass. The goal is an HONEST first pass:
capture what was said, and be explicit about what wasn't.

THE ARCHITECTURE (from a conversation with a WellSky engineer on how their
Scribe product handles this in production)
-----------------------------------------
Their first-pass completion sat around 40% of the target form. They did not chase
it with prompting. They built a second, interactive agent that identifies the gaps
and asks the user to fill them. The key detail:

    "all our forms are standardized, so we created a state object to represent
     the document. The agent is able to easily identify where fields are missing
     in a json format."

The gap detection is done by the SCHEMA, not the model.

    schema  ->  knows what a complete note contains       (a fixed checklist)
    model   ->  fills in what was actually said           (a language task)
    checker ->  compares the two, deterministically       (`if field is None`)
    model   ->  phrases a question about each gap         (a language task)
    human   ->  answers                                   (clinical judgment)

Note where the LLM is and isn't. It is NOT asked "what did you miss?" — that is
asking it to know what it doesn't know, which is the thing it is worst at; it will
produce a confident, plausible, wrong answer. It IS asked to phrase a question,
which is a pure language task it is good at.

**Deterministic where correctness matters. LLM where fluency matters.**

WHAT I AM CAREFUL NOT TO CLAIM
------------------------------
This does not "improve completeness". Of course completeness rises when a human
supplies the missing information — that is arithmetic, not achievement. Taking
credit for the human's answer would be dishonest.

The claim is narrower and it is the only one the design supports:

    the system correctly identifies WHICH fields are missing

That is what gets measured (precision/recall of gap detection). Everything after
that is the human doing the work.

V1 LIMITS, STATED PLAINLY
-------------------------
- Detects EMPTY fields, not THIN ones. "Is this null?" is a fact. "Is this
  section clinically adequate?" is a judgment, and a length threshold is a bad
  proxy for it — two lines may be complete for a cold and grossly inadequate for
  a cardiac workup. Deciding sufficiency needs a clinician, not a heuristic.
- The required-field set is a rough core, not a validated clinical standard. Which
  fields are mandatory for which encounter type is a clinical question I am not
  qualified to answer.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from soap_agent.schema_granular import (      # noqa: E402
    GranularSOAPNote, required_fields, field_description,
)


# ------------------------------------------------------------------- gap report

@dataclass
class Gap:
    """One required field the note does not contain."""
    section: str
    field: str
    description: str

    @property
    def path(self) -> str:
        return f"{self.section}.{self.field}"


@dataclass
class CoverageReport:
    """The deterministic output of the coverage check.

    Serialisable to JSON on purpose: this is the state object a UI would consume
    to highlight empty fields, exactly as described for the production system.
    """
    total_required: int
    captured: list[str] = field(default_factory=list)
    gaps: list[Gap] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        if self.total_required == 0:
            return 1.0
        return len(self.captured) / self.total_required

    @property
    def complete(self) -> bool:
        return not self.gaps

    def to_json(self) -> str:
        return json.dumps({
            "coverage": round(self.coverage, 3),
            "complete": self.complete,
            "captured": self.captured,
            "gaps": [{"path": g.path, "description": g.description}
                     for g in self.gaps],
        }, indent=2)


# --------------------------------------------------------- the deterministic core

def coverage_check(note: GranularSOAPNote) -> CoverageReport:
    """Compare the note against the schema's required fields.

    This is the whole idea, and it is deliberately trivial: a loop and a null
    check. No model, no judgment, no threshold, nothing to tune. It cannot
    hallucinate a gap and it cannot miss one, because the checklist is the schema
    and the test is `is None`.

    The unglamorousness is the feature. The part of the system that has to be
    right is the part with no judgment in it.
    """
    report = CoverageReport(total_required=0)

    for section_name, field_name in required_fields():
        report.total_required += 1
        section = getattr(note, section_name)
        value = getattr(section, field_name, None)

        # None means "not captured". An empty/whitespace string is the same thing —
        # a model that writes "" or "N/A" has not captured it either.
        missing = value is None or not str(value).strip() or \
            str(value).strip().lower() in {"n/a", "none", "not discussed", "unknown"}

        if missing:
            report.gaps.append(Gap(
                section=section_name,
                field=field_name,
                description=field_description(section_name, field_name),
            ))
        else:
            report.captured.append(f"{section_name}.{field_name}")

    return report


# ------------------------------------------------- the LLM's actual job: phrasing

_FALLBACK_TEMPLATES = {
    "temperature": "What was the patient's temperature?",
    "blood_pressure": "What was the patient's blood pressure?",
    "heart_rate": "What was the patient's heart rate?",
    "respiratory_rate": "What was the patient's respiratory rate?",
    "oxygen_saturation": "What was the oxygen saturation?",
    "exam_findings": "What did the physical exam show?",
    "general_appearance": "How did the patient appear on general inspection?",
    "chief_complaint": "What was the main reason for today's visit?",
    "history_present_illness": "How did this problem develop?",
    "duration": "How long has the patient had these symptoms?",
    "associated_symptoms": "Were there any other symptoms?",
    "denies": "What symptoms did the patient explicitly deny?",
    "relevant_history": "Any relevant past medical history or medications?",
    "primary_diagnosis": "What is the primary diagnosis?",
    "reasoning": "What is the clinical reasoning for this diagnosis?",
    "treatment": "What treatment was prescribed?",
    "patient_instructions": "What was the patient instructed to do?",
    "follow_up": "When should the patient follow up?",
    "referrals": "Were any referrals made?",
    "tests_ordered": "Were any tests or labs ordered?",
}


def clarification_questions(report: CoverageReport, transcript: str = "",
                            *, offline: bool = True) -> list[dict]:
    """Turn each gap into a question for the clinician.

    THIS is where the LLM belongs — and only here. Phrasing a natural question is
    a language task. Deciding WHICH fields are missing was already settled
    deterministically, above, and the model has no say in it.

    offline=True uses templates. That is not a lesser mode; for a fixed schema the
    templates are perfectly serviceable, and they cost nothing. The LLM version
    earns its keep by making the question specific to the encounter ("You mentioned
    starting Acetaminophen — how often should she take it?") rather than generic.
    """
    if not report.gaps:
        return []

    if offline:
        return [{
            "path": g.path,
            "question": _FALLBACK_TEMPLATES.get(g.field, f"What is the {g.description}?"),
        } for g in report.gaps]

    return _llm_questions(report, transcript)


def _llm_questions(report: CoverageReport, transcript: str) -> list[dict]:
    try:
        import anthropic
    except ImportError:
        return clarification_questions(report, offline=True)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return clarification_questions(report, offline=True)

    gaps_desc = "\n".join(f"- {g.path}: {g.description}" for g in report.gaps)
    prompt = (
        "A clinical note was drafted from the conversation below, but these "
        "required fields could not be filled from what was said.\n\n"
        f"CONVERSATION:\n{transcript[:4000]}\n\n"
        f"MISSING FIELDS:\n{gaps_desc}\n\n"
        "Write ONE short question per missing field, addressed to the clinician, "
        "to fill that gap. Ground each question in the encounter where you can "
        "(reference the actual complaint or medication) rather than asking "
        "generically.\n\n"
        "Return ONLY a JSON array: "
        '[{"path": "objective.temperature", "question": "..."}]. '
        "No markdown, no preamble."
    )
    try:
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model="claude-sonnet-4-6", max_tokens=900,
            messages=[{"role": "user", "content": prompt}])
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
        out = json.loads(text)
        valid = {g.path for g in report.gaps}
        # the model does not get to invent gaps — filter to the ones the checker found
        return [q for q in out if q.get("path") in valid]
    except Exception:
        return clarification_questions(report, offline=True)


# ------------------------------------------------------ applying human answers

def apply_answers(note: GranularSOAPNote, answers: dict[str, str]) -> GranularSOAPNote:
    """Write the clinician's answers back into the state object.

    Straight assignment. The human's words go in verbatim — no model rephrasing,
    no 'improving' what they wrote. If a clinician types "no exudate", the note
    says "no exudate".
    """
    updated = note.model_copy(deep=True)
    for path, value in answers.items():
        if not value or not str(value).strip():
            continue
        section_name, _, field_name = path.partition(".")
        section = getattr(updated, section_name, None)
        if section is not None and hasattr(section, field_name):
            setattr(section, field_name, str(value).strip())
    return updated
