#!/usr/bin/env python3
"""
Groundedness, done properly — separate workflows, flags not suppression.

WHAT I GOT WRONG
----------------
When the clinician filled the gaps, faithfulness DROPPED. The note had been
corrected by a doctor and my metric scored it worse. I treated that as a bug and
"fixed" it by making clinician-supplied content count as automatically supported.

That was wrong. I turned off an alarm that was working.

A WellSky engineer walked me through why, using a case from their production system:

    A form field asks: "complications resulting from the primary diagnosis"
    (question code A0112). In conversation, the clinician mentions comorbid
    lethargy stemming from the patient's diabetes. The model does not make the
    connection, because it has recorded the primary diagnosis as kidney disease,
    not diabetes. It can recite that part of the transcript, but it cannot infer
    that the answer to A0112 is lethargy. The clinician has to tell it.

Now: is "lethargy" grounded in the transcript?

**No. And that is the correct answer.** Nobody said "the complication of the primary
diagnosis is lethargy." The clinician derived it, using medical knowledge and
context the model does not have. It is a clinical inference, not a transcription.

So when the groundedness check flags it, the check is RIGHT. My provenance patch
suppressed exactly the signal a clinical system most needs to surface:

    "a human asserted this, and it is not in the recording — confirm they meant to"

That is not an error. It is an audit trail.

THE CORRECTED ARCHITECTURE
--------------------------
Two rules, both from the same conversation:

  1. SEPARATE THE WORKFLOWS.
     Extraction and clarification are different events. Groundedness is measured
     on the extraction — on what the model claimed the transcript said. The
     interactive portion is not reliant on the transcript at all; it is driven by
     the user's clinical judgment. Scoring them with one metric conflates a
     transcription task with an inference task.

  2. WORK BACKWARDS FROM THE RESULT.
     Do not score each stage as it runs. Let both workflows complete, then judge
     the final note against the ORIGINAL transcript. Whatever is not grounded gets
     FLAGGED — not deleted, not excused, not silently passed.

     Un-grounded clinician content is expected. It gets surfaced for confirmation,
     not suppressed.

WHAT THIS FILE PRODUCES
-----------------------
Not a single score. A per-field provenance ledger:

    GROUNDED             model extracted it, and it is traceable to the transcript
    UNGROUNDED_MODEL     model asserted it, and it is NOT in the transcript
                         -> this is a hallucination. The serious failure.
    UNGROUNDED_CLINICIAN clinician supplied it, and it is not in the transcript
                         -> EXPECTED. Flag for confirmation, do not treat as error.

Note that the two "ungrounded" categories are the same lexical failure and
completely different clinical events. A single faithfulness number cannot tell them
apart — which is precisely why a single faithfulness number was never the right
output.

Run:
    python scripts/groundedness.py
    python scripts/groundedness.py --judge     # LLM judge instead of lexical
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from enum import Enum

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from data.synthetic import generate_encounter                        # noqa: E402
from soap_agent.guardrails import redact_transcript                  # noqa: E402
from soap_agent.schema_granular import GranularSOAPNote              # noqa: E402
from soap_agent.coverage import coverage_check, apply_answers        # noqa: E402


BAR = "=" * 76
_WORD = re.compile(r"[a-z0-9]+")


class Source(str, Enum):
    """Where a field's content came from. The system always knew this."""
    MODEL = "model"          # extracted from the transcript by the LLM
    CLINICIAN = "clinician"  # supplied by a human answering a clarification


class Provenance(str, Enum):
    """WHERE the content came from. A fact about origin, not about quality.

    This is deliberately separate from groundedness. 'The clinician added this' and
    'this is not in the transcript' are two different facts, and a clinical record
    needs to answer them independently:

        "show me everything not traceable to the conversation"  -> filter groundedness
        "show me everything the clinician contributed"          -> filter provenance

    Mash them into one label and neither question can be asked cleanly.
    """
    TRANSCRIPT = "transcript"   # the model extracted it from the conversation
    CLINICIAN = "clinician"     # a human supplied it during clarification
    UNKNOWN = "unknown"         # neither — this is the alarming case


@dataclass
class FieldAudit:
    """Two independent axes, never collapsed into one verdict.

    groundedness  — is this content supported by the ORIGINAL transcript? (bool)
    provenance    — where did it come from? (transcript / clinician / unknown)

    The whole point of keeping them apart:

        grounded=False, provenance=clinician  -> intentional clinical input. FLAG for
                                                 confirmation. NOT a hallucination.
        grounded=False, provenance=unknown    -> content from no valid source. This is
                                                 the model hallucinating. SERIOUS.

    Same groundedness value (False). Completely different clinical events. Only the
    provenance axis tells them apart — which is exactly why it must be its own field.
    """
    path: str
    value: str
    grounded_to_transcript: bool
    provenance: Provenance
    score: float

    @property
    def attention_required(self) -> bool:
        # anything not grounded in the transcript needs a human to confirm intent,
        # regardless of who added it. grounded content is fine.
        return not self.grounded_to_transcript

    @property
    def is_hallucination(self) -> bool:
        # the one combination that is a real defect: not in the transcript AND
        # not attributable to a clinician. content from nowhere.
        return (not self.grounded_to_transcript
                and self.provenance is Provenance.UNKNOWN)


# ------------------------------------------------------------------ grounding

_STOP = {"the","and","for","with","that","this","was","were","has","have","had",
         "not","you","your","are","but","from","who","which","will","would","can",
         "should","been","being","also","any","all","its","there","here","then",
         "than","some","did","does","get","got","just","like","about","into","over",
         "out","off","patient","reports","report","states","note","noted","today",
         "per","plus","mg","daily","needed"}


def _terms(text: str) -> set[str]:
    return {t for t in _WORD.findall(str(text).lower())
            if t not in _STOP and len(t) > 2}


def grounded_lexical(value: str, transcript: str) -> float:
    """Share of the field's content terms that appear in the transcript.

    Lexical, and therefore blunt: it cannot tell that "nasal congestion" and
    "stuffy nose" are the same clinical fact. That limitation is why --judge exists.
    """
    terms = _terms(value)
    if not terms:
        return 1.0
    src = set(_WORD.findall(transcript.lower()))
    return sum(1 for t in terms if t in src) / len(terms)


def grounded_judge(value: str, transcript: str) -> float:
    """Ask an LLM whether the claim is supported by the transcript.

    Handles paraphrase, which the lexical scorer cannot. Note the judge is asked
    ONLY about the transcript — not about whether the claim is clinically sensible.
    Those are different questions and conflating them is how you end up with a judge
    that approves plausible fabrications.
    """
    try:
        import anthropic
    except ImportError:
        return grounded_lexical(value, transcript)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return grounded_lexical(value, transcript)

    prompt = (
        "Decide whether a statement from a clinical note is supported by the "
        "conversation it was drawn from.\n\n"
        f"CONVERSATION:\n{transcript[:5000]}\n\n"
        f"STATEMENT:\n\"{value}\"\n\n"
        "SUPPORTED means the conversation contains this information, including as "
        "a clinical paraphrase (e.g. 'denies fever' when the patient said no to "
        "fever; 'nasal congestion' for 'stuffy nose').\n"
        "UNSUPPORTED means the conversation does not contain it — it was inferred "
        "or introduced from outside.\n\n"
        "Judge ONLY against the conversation. Do not judge whether the statement is "
        "clinically reasonable. A plausible statement that was never discussed is "
        "UNSUPPORTED.\n\n"
        "Answer with exactly one word: SUPPORTED or UNSUPPORTED."
    )
    try:
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model="claude-sonnet-4-6", max_tokens=8,
            messages=[{"role": "user", "content": prompt}])
        text = "".join(b.text for b in resp.content if b.type == "text").upper()
        return 0.0 if "UNSUPPORTED" in text else 1.0
    except Exception:
        return grounded_lexical(value, transcript)


# ---------------------------------------------------------------- the audit

def audit(note: GranularSOAPNote, transcript: str,
          clinician_fields: set[str], *, use_judge: bool = False,
          threshold: float = 0.5) -> list[FieldAudit]:
    """Judge the FINAL note against the ORIGINAL transcript.

    Working backwards from the result, as advised — not scoring each stage as it
    runs. Both workflows have completed; this asks a single question of the output:
    what in this note is not traceable to the recording, and who put it there?
    """
    scorer = grounded_judge if use_judge else grounded_lexical
    out: list[FieldAudit] = []

    for section_name in ("subjective", "objective", "assessment", "plan"):
        section = getattr(note, section_name)
        for fname in type(section).model_fields:
            value = getattr(section, fname)
            if not value:
                continue

            path = f"{section_name}.{fname}"
            score = scorer(str(value), transcript)
            grounded = score >= threshold

            # provenance is a SEPARATE fact from groundedness. we know who added
            # each field independently of whether it's in the transcript.
            if path in clinician_fields:
                prov = Provenance.CLINICIAN
            elif grounded:
                prov = Provenance.TRANSCRIPT      # model extracted it AND it checks out
            else:
                prov = Provenance.UNKNOWN         # model asserted it, not in transcript

            out.append(FieldAudit(
                path=path, value=str(value),
                grounded_to_transcript=grounded,
                provenance=prov, score=score))

    return out


def report(audits: list[FieldAudit]) -> None:
    # the two axes, queried INDEPENDENTLY — the whole reason for splitting them
    grounded = [a for a in audits if a.grounded_to_transcript]
    not_grounded = [a for a in audits if not a.grounded_to_transcript]
    clinician_added = [a for a in audits if a.provenance is Provenance.CLINICIAN]
    hallucinations = [a for a in audits if a.is_hallucination]

    print(f"\n{BAR}")
    print("  AUDIT — final note vs. ORIGINAL transcript")
    print("  two independent axes: groundedness (is it in the transcript?)")
    print("                        provenance   (who put it there?)")
    print(BAR)
    print(f"    fields populated            {len(audits)}")
    print(f"    grounded to transcript      {len(grounded)}")
    print(f"    NOT grounded to transcript  {len(not_grounded)}")
    print(f"      of those, clinician-added {len([a for a in not_grounded if a.provenance is Provenance.CLINICIAN])}   (flag, expected)")
    print(f"      of those, from nowhere    {len(hallucinations)}   (HALLUCINATION)")
    print()

    if hallucinations:
        print("  !! NOT GROUNDED, provenance=UNKNOWN — model asserted content from no source")
        print("  " + "-" * 72)
        for a in hallucinations:
            print(f"     {a.path:<34} {a.value[:44]}")
        print("     The serious failure: not in the transcript, not from a clinician.")
        print()

    flagged = [a for a in not_grounded if a.provenance is Provenance.CLINICIAN]
    if flagged:
        print("  ?  NOT GROUNDED, provenance=CLINICIAN — confirm intent")
        print("  " + "-" * 72)
        for a in flagged:
            print(f"     {a.path:<34} {a.value[:44]}")
        print()
        print("     grounded_to_transcript = False   (correctly — it isn't in the transcript)")
        print("     provenance             = clinician")
        print("     attention_required     = True")
        print()
        print("     NOT a hallucination. The clinician contributed judgment the")
        print("     transcript doesn't contain. The record simply notes WHERE it came")
        print("     from, so it's traceable later. That is provenance, not a defect.")
        print()

    print("  " + "-" * 72)
    print("  Why there is no single 'faithfulness score' here:")
    print()
    print("    An ungrounded model claim and an ungrounded clinician claim are the")
    print("    SAME lexical failure and COMPLETELY DIFFERENT clinical events. One is")
    print("    a hallucination. The other is a doctor doing their job. Averaging them")
    print("    into one number destroys the only distinction that matters.")
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", action="store_true",
                    help="use an LLM judge (handles paraphrase; needs API key)")
    args = ap.parse_args()

    enc = generate_encounter(seed=0)
    transcript = redact_transcript(
        enc.transcript, known_names=[enc.patient_name, enc.provider_name]).text

    # --- WORKFLOW 1: extraction (this is what groundedness is really about) ---
    note = GranularSOAPNote()
    t = transcript.lower()
    note.subjective.chief_complaint = "Sore throat"
    note.subjective.history_present_illness = (
        "Sore throat with fever, nasal congestion, and cough.")
    note.subjective.duration = "Three days"
    m = re.search(r"temp\s*([\d.]+)\s*f", t)
    if m:
        note.objective.temperature = f"{m.group(1)} F"
    note.objective.exam_findings = "Pharynx erythematous."
    note.assessment.primary_diagnosis = "Acute viral upper respiratory infection"
    note.plan.treatment = "Acetaminophen 500 mg every 6 hours as needed"

    before = coverage_check(note)

    # --- WORKFLOW 2: clarification (separate event, NOT transcript-driven) ----
    clinician_answers = {
        "plan.patient_instructions": "Rest and increase fluid intake.",
        "plan.follow_up": "Return if symptoms worsen or persist beyond 10 days.",
        # the A0112-style case: a clinical inference the model could not make.
        # nobody said this in the conversation. the clinician derived it.
        "assessment.reasoning":
            "Viral etiology likely given absence of tonsillar exudate.",
    }
    final = apply_answers(note, clinician_answers)
    after = coverage_check(final)

    # split what the clinician supplied into the two categories that were being
    # confusingly summed: required gaps that got filled, vs optional fields the
    # clinician volunteered (the A0112-style demonstration case).
    required_paths = {f"{s}.{f}" for s, f in
                      __import__("soap_agent.schema_granular",
                                 fromlist=["required_fields"]).required_fields()}
    filled_required = [p for p in clinician_answers if p in required_paths]
    volunteered_optional = [p for p in clinician_answers if p not in required_paths]

    print(f"\n{BAR}")
    print("  TWO SEPARATE WORKFLOWS")
    print(BAR)
    print("    WORKFLOW 1 — extraction (model reads the transcript)")
    print(f"        required-field coverage before clarification : "
          f"{len(before.captured)}/{before.total_required}")
    print()
    print("    WORKFLOW 2 — clarification (clinician answers, NOT transcript-driven)")
    print(f"        required gaps completed      : "
          f"{len(filled_required)}/{len(before.gaps)}")
    print(f"        optional fields volunteered  : {len(volunteered_optional)}   "
          f"(A0112-style clinician inference)")
    print(f"        total clinician-supplied     : {len(clinician_answers)}")
    print()
    print(f"    AFTER clarification")
    print(f"        required-field coverage      : "
          f"{len(after.captured)}/{after.total_required}")
    print()
    print("    These are different events. The second is NOT transcript-driven —")
    print("    it runs on the clinician's judgment. Scoring them with one metric")
    print("    conflates a transcription task with an inference task.")

    audits = audit(final, transcript, set(clinician_answers),
                   use_judge=args.judge)
    report(audits)

    print(BAR)
    print("  WHAT CHANGED, AND WHY")
    print(BAR)
    print("""
    BEFORE:  clinician answers were forced to count as "supported", so the
             groundedness check passed. I had suppressed the failure.

    NOW:     clinician answers that are not in the transcript FAIL the check, and
             are surfaced as flags rather than errors.

    The failure was never a bug. It was the system correctly reporting that a
    human had asserted something the recording does not contain. In a clinical
    record that is exactly what you want to know — and I had switched it off.
""")


if __name__ == "__main__":
    main()
