"""
Evidence-surfaced clarification — draw the clinician's attention, defer the judgment.

WHAT WAS MISSING
----------------
The coverage layer detects a gap and asks a bare question:

    "What complications resulted from the primary diagnosis?"

The WellSky engineer described something richer. The agent should scan the
document, identify the incomplete field, and *draw the user's attention to
relevant information* — then let the clinician determine the meaning:

    "The agent can quickly scan the document for fields left incomplete and draw
     your attention to relevant information faster than you can find it, but defer
     to you to comprehend it for completeness."

His worked example is the whole reason this matters:

    Field A0112 asks for "complications resulting from the primary diagnosis".
    In conversation, the clinician mentioned comorbid lethargy from the patient's
    diabetes. But the model recorded the primary diagnosis as kidney disease, not
    diabetes — so it cannot connect "lethargy" to A0112. It can RECITE that part
    of the transcript. It cannot INTERPRET it. The clinician has to.

So the interaction should not be a blank question. It should surface the relevant
transcript line AND the question, and let the clinician decide:

    Missing field:      Complications from primary diagnosis
    Relevant evidence:  "...the lethargy is related to her diabetes..."
    Current primary dx: Kidney disease
    Question:           Should lethargy be recorded as a complication here?
    -> clinician confirms / corrects / rejects

THE DIVISION OF LABOUR (unchanged, and this is the point)
---------------------------------------------------------
    schema     detects the gap                      (deterministic)
    retriever  finds possibly-relevant transcript   (model — a SEARCH task)
    model      phrases the question                 (model — a LANGUAGE task)
    clinician  decides what it means                (human — the JUDGMENT)

The model is allowed to LOCATE and SUMMARISE. It is never allowed to DECIDE. It
surfaces "here is a line that might be relevant"; the clinician rules on whether it
answers the field. Surfacing is a search problem. Interpreting is a clinical one.
Keeping that line sharp is the entire safety argument.

WHY THE MODEL MUST NOT DECIDE HERE
----------------------------------
The A0112 case is exactly where a model goes wrong: it sees "lethargy" and
"diabetes" in the transcript, but it has the wrong primary diagnosis, so if it
were allowed to auto-fill, it would either miss the connection or invent a wrong
one. The clinician has context the model does not. So the model's job stops at
"this line looks relevant to this field — you decide."
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from soap_agent.coverage import CoverageReport, Gap                  # noqa: E402


# ------------------------------------------------------- the interaction object

@dataclass
class ClarificationItem:
    """Everything the clinician needs to resolve ONE gap.

    This is the state object for a single interaction — deliberately richer than a
    bare question, but the design principle is that the system should surface evidence,
    not just ask.
    """
    field_id: str
    field_name: str
    status: str = "missing"
    relevant_transcript_evidence: list[str] = field(default_factory=list)
    clarification_question: str = ""
    clinician_answer: str | None = None
    confirmed_by_clinician: bool = False

    def to_dict(self) -> dict:
        return {
            "field_id": self.field_id,
            "field_name": self.field_name,
            "status": self.status,
            "relevant_transcript_evidence": self.relevant_transcript_evidence,
            "clarification_question": self.clarification_question,
            "clinician_answer": self.clinician_answer,
            "confirmed_by_clinician": self.confirmed_by_clinician,
        }


# ----------------------------------------------------- evidence surfacing

_SENT_SPLIT = re.compile(r"(?<=[.?!])\s+|\n")


def _sentences(transcript: str) -> list[str]:
    out = []
    for line in _SENT_SPLIT.split(transcript):
        line = re.sub(r"^(provider|patient|doctor|dr|nurse)\s*:\s*", "",
                      line.strip(), flags=re.I)
        if len(line) > 8:
            out.append(line)
    return out


def find_evidence_lexical(gap: Gap, transcript: str, k: int = 2) -> list[str]:
    """Surface transcript lines that MIGHT be relevant to a missing field.

    Deliberately high-recall and low-precision: it is better to show the clinician
    one extra irrelevant line than to hide the relevant one. The clinician filters;
    the system's job is only to save them the search. Missing the relevant line is
    the expensive error — that is the whole value being offered.

    Lexical here (keyword overlap between the field's meaning and each sentence).
    A production version would embed both — the same swap made in the retriever.
    """
    cue_terms = set(re.findall(r"[a-z]{4,}", gap.description.lower()))
    cue_terms |= set(re.findall(r"[a-z]{4,}", gap.field.replace("_", " ").lower()))

    scored = []
    for sent in _sentences(transcript):
        words = set(re.findall(r"[a-z]{4,}", sent.lower()))
        overlap = len(cue_terms & words)
        if overlap:
            scored.append((overlap, sent))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in scored[:k]]


def find_evidence_semantic(gap: Gap, transcript: str, k: int = 2) -> list[str]:
    """Surface relevant lines by MEANING, not keyword overlap.

    This is why the lexical version is not enough, and it is the A0112 case exactly:
    the relevant line ("lethargy... related to diabetes") shares NO words with the
    field name ("complications from primary diagnosis"). Keyword matching cannot
    connect them. Embeddings can, because "lethargy" and "complications" are close
    in meaning even with zero shared characters.

    Same lesson as the retriever: in clinical language, the words the patient uses
    and the words the form uses are rarely the same words.
    """
    try:
        from sentence_transformers import SentenceTransformer, util
    except ImportError:
        return find_evidence_lexical(gap, transcript, k)
    try:
        global _EV_MODEL
        if _EV_MODEL is None:
            _EV_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        query = (f"complications, comorbidities, or secondary problems related to "
                 f"the diagnosis. {gap.description}. {gap.field.replace('_', ' ')}")
        sents = _sentences(transcript)
        if not sents:
            return []
        q = _EV_MODEL.encode(query, convert_to_tensor=True, normalize_embeddings=True)
        s = _EV_MODEL.encode(sents, convert_to_tensor=True, normalize_embeddings=True)
        sims = util.cos_sim(q, s)[0]
        ranked = sorted(zip(sims.tolist(), sents), reverse=True)
        # keep only lines with real similarity — no evidence beats wrong evidence
        return [sent for score, sent in ranked[:k] if score >= 0.12]
    except Exception:
        return find_evidence_lexical(gap, transcript, k)


_EV_MODEL = None


def build_clarifications(report: CoverageReport, transcript: str,
                         *, offline: bool = True) -> list[ClarificationItem]:
    """One interaction object per gap: evidence + question, ready for the clinician.

    Surfacing (finding evidence) and phrasing (writing the question) are both model
    tasks. Deciding is the clinician's. This function assembles the first two and
    leaves clinician_answer / confirmed_by_clinician empty for the human to fill.
    """
    items: list[ClarificationItem] = []
    for gap in report.gaps:
        evidence = find_evidence_semantic(gap, transcript)
        question = _phrase_question(gap, evidence, offline=offline)
        items.append(ClarificationItem(
            field_id=gap.path,
            field_name=gap.description,
            relevant_transcript_evidence=evidence,
            clarification_question=question,
        ))
    return items


_TEMPLATES = {
    "patient_instructions": "What was the patient instructed to do?",
    "follow_up": "When should the patient follow up?",
    "temperature": "What was the patient's temperature?",
    "exam_findings": "What did the physical exam show?",
    "primary_diagnosis": "What is the primary diagnosis?",
    "treatment": "What treatment was prescribed?",
    "chief_complaint": "What was the main reason for the visit?",
    "history_present_illness": "How did the current problem develop?",
    "duration": "How long has the patient had these symptoms?",
}


def _phrase_question(gap: Gap, evidence: list[str], *, offline: bool) -> str:
    field_key = gap.path.split(".")[-1]
    if offline:
        return _TEMPLATES.get(field_key, f"What is the {gap.description}?")
    return _llm_question(gap, evidence)


def _llm_question(gap: Gap, evidence: list[str]) -> str:
    try:
        import anthropic
    except ImportError:
        return _phrase_question(gap, evidence, offline=True)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return _phrase_question(gap, evidence, offline=True)

    ev = "\n".join(f"  - {e}" for e in evidence) or "  (none found)"
    prompt = (
        f"A clinical note is missing this field: {gap.description} ({gap.path}).\n\n"
        f"Possibly-relevant lines from the conversation:\n{ev}\n\n"
        "Write ONE short question for the clinician to resolve this field. If the "
        "evidence above looks relevant, reference it so the clinician can confirm "
        "or correct — but do NOT assert the answer yourself. You surface; the "
        "clinician decides.\n\n"
        "Return only the question text."
    )
    try:
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model="claude-sonnet-4-6", max_tokens=120,
            messages=[{"role": "user", "content": prompt}])
        return "".join(b.text for b in resp.content if b.type == "text").strip()
    except Exception:
        return _phrase_question(gap, evidence, offline=True)


def record_answer(item: ClarificationItem, answer: str,
                  confirmed: bool = True) -> ClarificationItem:
    """Write the clinician's decision back into the interaction object.

    Note `confirmed_by_clinician`: for a semantically ambiguous field like the
    A0112/lethargy case, the value should not be trusted until the clinician has
    explicitly confirmed it. Surfacing a suggestion is not the same as the clinician
    accepting it.
    """
    item.clinician_answer = answer.strip() if answer else None
    item.confirmed_by_clinician = confirmed and bool(answer and answer.strip())
    item.status = "resolved" if item.confirmed_by_clinician else "pending"
    return item
