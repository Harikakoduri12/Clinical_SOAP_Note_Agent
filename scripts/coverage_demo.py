#!/usr/bin/env python3
"""
Coverage + clarification demo — the system knowing what it doesn't know.

Shows the full loop:

    1. first pass      model fills what the conversation actually supports
    2. coverage check  the SCHEMA (not the model) finds the empty required fields
    3. gap report      JSON — the state object a UI would consume
    4. questions       the model phrases one question per gap
    5. human answers   simulated here; a clinician in reality
    6. completed note  answers written back into the state object

Run:
    python scripts/coverage_demo.py              # templates, no API key needed
    python scripts/coverage_demo.py --live       # LLM extraction + LLM questions
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from data.synthetic import generate_encounter                      # noqa: E402
from soap_agent.schema_granular import GranularSOAPNote, required_fields  # noqa: E402
from soap_agent.coverage import (                                  # noqa: E402
    coverage_check, clarification_questions, apply_answers,
)
from soap_agent.guardrails import redact_transcript                # noqa: E402


BAR = "=" * 74


def first_pass(transcript: str, *, offline: bool) -> GranularSOAPNote:
    """Fill the granular schema from the conversation.

    The prompt says: leave a field NULL if the conversation does not support it.
    Do not guess. Do not write 'not discussed'. Just leave it null.

    That instruction is the whole point. A model told to 'be thorough' will invent
    a plausible temperature. A model told 'null is an acceptable answer' will admit
    it doesn't know — and then the checker can see the gap.
    """
    if offline:
        return _stub_first_pass(transcript)
    return _llm_first_pass(transcript)


def _stub_first_pass(transcript: str) -> GranularSOAPNote:
    """A deliberately partial extraction, so the demo runs without an API key.

    It captures the easy things and misses several required fields — which is
    realistic: the first pass in production sat around 40% of the target form.
    """
    note = GranularSOAPNote()
    t = transcript.lower()

    if "sore throat" in t:
        note.subjective.chief_complaint = "Sore throat"
        note.subjective.history_present_illness = (
            "Sore throat with fever, nasal congestion, and cough.")
    if "three days" in t or "3 days" in t:
        note.subjective.duration = "Three days"

    m = re.search(r"temp\s*([\d.]+)\s*f", t)
    if m:
        note.objective.temperature = f"{m.group(1)} F"

    if "erythematous" in t:
        note.objective.exam_findings = "Pharynx erythematous."

    if "upper respiratory" in t:
        note.assessment.primary_diagnosis = "Acute viral upper respiratory infection"

    if "acetaminophen" in t:
        note.plan.treatment = "Acetaminophen 500 mg every 6 hours as needed"

    # deliberately NOT filled: patient_instructions, follow_up
    # -> these become the gaps the checker will find
    return note


def _llm_first_pass(transcript: str) -> GranularSOAPNote:
    import anthropic

    schema_json = json.dumps(GranularSOAPNote.model_json_schema(), indent=2)
    prompt = (
        "Fill this clinical note schema from the conversation.\n\n"
        "CRITICAL RULE: if the conversation does not clearly support a field, "
        "leave it null. Do NOT guess. Do NOT infer. Do NOT write 'not discussed' "
        "or 'N/A' — just leave it null. A null field is a correct answer when the "
        "information was not stated. Inventing a plausible value is the worst "
        "thing you can do here.\n\n"
        f"SCHEMA:\n{schema_json}\n\n"
        f"CONVERSATION:\n{transcript}\n\n"
        "Return ONLY the JSON object matching the schema. No markdown, no preamble."
    )
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=2000,
        messages=[{"role": "user", "content": prompt}])
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    return GranularSOAPNote(**json.loads(text))


def show_note(note: GranularSOAPNote, title: str) -> None:
    """Print the note, distinguishing GAPS from fields that were simply not applicable.

    This distinction is the whole point and the display has to make it obvious:

        [ GAP - REQUIRED ]   an empty REQUIRED field. The note is incomplete.
                             The clinician gets asked about this.

        (not applicable)     an empty OPTIONAL field. Nothing was ordered, nothing
                             was measured. Empty is the CORRECT answer and nobody
                             gets asked.

    Showing both as "NOT CAPTURED" would be misleading — it implies six things are
    missing when only two are. And in a real UI, flagging every empty field would
    cause alert fatigue: the clinician gets four pointless questions ("were any
    referrals made?" - no), starts clicking through them, and stops reading the
    prompts that actually matter.
    """
    required = set(required_fields())

    print(f"\n{title}")
    print("-" * 74)
    for section_name in ("subjective", "objective", "assessment", "plan"):
        section = getattr(note, section_name)
        print(f"  {section_name.upper()}")
        for fname in type(section).model_fields:
            val = getattr(section, fname)
            is_required = (section_name, fname) in required

            if val:
                mark, shown = " OK ", str(val)[:60]
            elif is_required:
                mark, shown = "!GAP", "[ GAP - REQUIRED, will ask clinician ]"
            else:
                mark, shown = "    ", "(not applicable - nothing to record)"

            print(f"    {mark}  {fname:<26} {shown}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true",
                    help="use the Anthropic API for extraction and questions")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    offline = not args.live

    enc = generate_encounter(seed=args.seed)
    redaction = redact_transcript(
        enc.transcript, known_names=[enc.patient_name, enc.provider_name])
    redacted = redaction.text

    print(BAR)
    print("  COVERAGE + CLARIFICATION")
    print("  making the system know what it doesn't know")
    print(BAR)
    print(f"  encounter: {enc.encounter_id}   mode: "
          f"{'LIVE (Anthropic API)' if args.live else 'offline (templates)'}")

    # ---------------------------------------------------------------- 1. first pass
    note = first_pass(redacted, offline=offline)
    show_note(note, "[1] FIRST PASS  — model fills only what the conversation supports")

    # ------------------------------------------------------------ 2. coverage check
    report = coverage_check(note)
    print(f"\n[2] COVERAGE CHECK  — deterministic, no model involved")
    print("-" * 74)
    print(f"    required fields : {report.total_required}")
    print(f"    captured        : {len(report.captured)}")
    print(f"    gaps            : {len(report.gaps)}")
    print(f"    coverage        : {report.coverage:.0%}")
    print()
    print("    This ran with no LLM. The schema is the checklist; the test is")
    print("    `is None`. It cannot hallucinate a gap and it cannot miss one.")

    # ---------------------------------------------------------------- 3. gap report
    print(f"\n[3] GAP REPORT (JSON)  — the state object a UI would consume")
    print("-" * 74)
    for line in report.to_json().splitlines():
        print(f"    {line}")

    if report.complete:
        print("\n    No gaps. Nothing to ask.")
        return

    # ----------------------------------------------------------------- 4. questions
    questions = clarification_questions(report, redacted, offline=offline)
    print(f"\n[4] CLARIFICATION QUESTIONS  — the ONLY place the LLM is used here")
    print("-" * 74)
    for q in questions:
        print(f"    [{q['path']}]")
        print(f"      -> {q['question']}")
    print()
    print("    Note what the model was NOT asked: 'what did you miss?'. It has no")
    print("    idea. The checker already knows. The model only phrases the question.")

    # ------------------------------------------------------------- 5. human answers
    simulated = {
        "subjective.duration": "Three days",
        "objective.temperature": "100.2 F",
        "objective.exam_findings": "Pharynx erythematous, no tonsillar exudate.",
        "subjective.chief_complaint": "Sore throat",
        "subjective.history_present_illness":
            "Three days of sore throat, low-grade fever, nasal congestion, cough.",
        "assessment.primary_diagnosis": "Acute viral upper respiratory infection",
        "plan.treatment": "Acetaminophen 500 mg every 6 hours as needed",
        "plan.patient_instructions": "Rest and increase fluid intake.",
        "plan.follow_up": "Return if symptoms worsen or persist beyond 10 days.",
    }
    answers = {q["path"]: simulated.get(q["path"], "") for q in questions}
    answered = {k: v for k, v in answers.items() if v}

    print(f"\n[5] CLINICIAN ANSWERS  (simulated for the demo)")
    print("-" * 74)
    for path, val in answered.items():
        print(f"    {path:<36} {val[:50]}")

    # ------------------------------------------------------------- 6. completed note
    completed = apply_answers(note, answered)
    final = coverage_check(completed)
    show_note(completed, "[6] COMPLETED NOTE  — human answers written into the state")

    print(f"\n{BAR}")
    print("  RESULT")
    print(BAR)
    print(f"    coverage before : {report.coverage:.0%}  ({len(report.gaps)} gaps)")
    print(f"    coverage after  : {final.coverage:.0%}  ({len(final.gaps)} gaps)")
    print()
    print("  What this demo does NOT claim:")
    print("    That the SYSTEM improved completeness. It didn't. A human supplied")
    print("    the missing information — that is arithmetic, not achievement, and")
    print("    taking credit for it would be dishonest.")
    print()
    print("  What it DOES claim:")
    print("    The system correctly identified WHICH fields were missing, without")
    print("    asking a model to know what it didn't know. That is the measurable")
    print("    contribution, and it is the only one the design supports.")
    print()


if __name__ == "__main__":
    main()
