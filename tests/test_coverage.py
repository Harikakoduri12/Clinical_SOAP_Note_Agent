"""
Coverage-detection tests.

The claim under test is narrow and specific:

    the system correctly identifies WHICH required fields are missing

NOT "the system improves completeness" — a human supplies the missing information,
so completeness rising is arithmetic, not achievement. Only gap DETECTION is the
system's contribution, so gap detection is what gets tested.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from soap_agent.schema_granular import GranularSOAPNote, required_fields
from soap_agent.coverage import (
    coverage_check, clarification_questions, apply_answers,
)


def _full_note() -> GranularSOAPNote:
    """A note with every required field populated."""
    n = GranularSOAPNote()
    n.subjective.chief_complaint = "Sore throat"
    n.subjective.history_present_illness = "Three days of sore throat and fever."
    n.subjective.duration = "Three days"
    n.objective.temperature = "100.2 F"
    n.objective.exam_findings = "Pharynx erythematous."
    n.assessment.primary_diagnosis = "Acute viral URI"
    n.plan.treatment = "Acetaminophen 500 mg q6h PRN"
    n.plan.patient_instructions = "Rest, fluids."
    n.plan.follow_up = "Return if worse after 10 days."
    return n


# ------------------------------------------------------------ the core property

def test_complete_note_has_no_gaps():
    report = coverage_check(_full_note())
    assert report.complete
    assert report.gaps == []
    assert report.coverage == 1.0


def test_empty_note_flags_every_required_field():
    report = coverage_check(GranularSOAPNote())
    assert len(report.gaps) == len(required_fields())
    assert report.coverage == 0.0


@pytest.mark.parametrize("section,field", required_fields())
def test_each_required_field_is_detected_when_missing(section, field):
    """Remove exactly one required field; the checker must find exactly that one.

    Parametrised over the schema itself, so adding a required field automatically
    adds a test for it. The checklist and the tests cannot drift apart.
    """
    note = _full_note()
    setattr(getattr(note, section), field, None)

    report = coverage_check(note)
    paths = [g.path for g in report.gaps]

    assert paths == [f"{section}.{field}"], (
        f"expected exactly ['{section}.{field}'] to be flagged, got {paths}"
    )


# --------------------------------------------------- the sneaky failure mode

@pytest.mark.parametrize("evasion", ["", "   ", "N/A", "None", "not discussed",
                                     "unknown", "NOT DISCUSSED"])
def test_evasive_placeholder_counts_as_missing(evasion):
    """A model that writes 'N/A' has NOT captured the field — it has hidden the gap.

    This is the failure that matters most. If the checker accepted "N/A" as a
    value, the field would count as captured, the gap would never be reported, and
    the clinician would never be asked. The note would look complete and be silently
    incomplete — the exact danger this whole design exists to prevent.
    """
    note = _full_note()
    note.objective.temperature = evasion

    report = coverage_check(note)
    assert "objective.temperature" in [g.path for g in report.gaps], (
        f"'{evasion}' must be treated as a gap, not as a captured value"
    )


# ---------------------------------------------------- questions follow the gaps

def test_questions_are_generated_only_for_gaps():
    note = _full_note()
    note.plan.follow_up = None
    note.objective.temperature = None

    report = coverage_check(note)
    questions = clarification_questions(report, offline=True)

    asked = {q["path"] for q in questions}
    assert asked == {"plan.follow_up", "objective.temperature"}


def test_no_gaps_means_no_questions():
    report = coverage_check(_full_note())
    assert clarification_questions(report, offline=True) == []


# ------------------------------------------------------- answers close the loop

def test_answers_fill_the_gaps_verbatim():
    """The clinician's words go in unchanged. No model rephrasing them."""
    note = _full_note()
    note.plan.follow_up = None

    before = coverage_check(note)
    assert not before.complete

    after_note = apply_answers(note, {"plan.follow_up": "Two weeks."})
    after = coverage_check(after_note)

    assert after.complete
    assert after_note.plan.follow_up == "Two weeks."


def test_apply_answers_does_not_mutate_the_original():
    note = _full_note()
    note.plan.follow_up = None
    apply_answers(note, {"plan.follow_up": "Two weeks."})
    assert note.plan.follow_up is None, "original note must not be mutated"


def test_blank_answer_leaves_the_gap_open():
    """A clinician who skips a question has not answered it. The gap stays."""
    note = _full_note()
    note.plan.follow_up = None

    after_note = apply_answers(note, {"plan.follow_up": "   "})
    assert not coverage_check(after_note).complete


# ------------------------------------------------------------ determinism

def test_coverage_check_is_deterministic():
    """Same note, same answer, every time. No model, no randomness, no threshold."""
    note = _full_note()
    note.objective.temperature = None
    runs = [tuple(g.path for g in coverage_check(note).gaps) for _ in range(20)]
    assert len(set(runs)) == 1, "coverage_check must be deterministic"
