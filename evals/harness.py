"""
Evaluation harness.

Scores generated notes against the gold set. Two tiers of check, deliberately:

  DETERMINISTIC (strict, non-gameable, no AI):
    - structural_validity : did it pass the Pydantic schema?
    - phi_leakage         : did any planted PHI survive?  (must be 0)
    - icd_grounding       : is every emitted code in the retrieved set?
    - completeness        : recall of gold clinical facts (entity overlap)
    - code_accuracy       : does the emitted ICD match the gold ICD?

  LLM-AS-JUDGE (for the fuzzy part only):
    - faithfulness        : is everything in the note supported by the source?

We lean on deterministic checks for everything checkable and reserve the judge
for genuine semantic judgement. The judge is itself fallible, so the harness
also supports spot-checking judge scores against human labels (see
`judge_agreement`). That "distrust your own grader" step is the point.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from soap_agent.schema import validate_draft          # noqa: E402
from soap_agent.guardrails import detect_phi_leak      # noqa: E402


_WORD = re.compile(r"[a-z0-9]+")


def _facts(note: dict) -> set[str]:
    """Extract a bag of clinical 'facts' (content tokens) from a note's fields.

    Simple, transparent operationalization: significant tokens from subjective,
    objective, assessment, plan, and medication names. Stopwords removed.
    """
    stop = {"the", "and", "for", "with", "reports", "patient", "of", "to", "a",
            "an", "in", "on", "or", "if", "as", "at", "is", "no", "past",
            "over", "here", "you", "your", "i", "ve", "that", "s", "me",
            "about", "been", "will", "be", "not", "this", "based", "it",
            "look", "like", "let", "check", "vitals", "d", "start", "okay",
            "sounds", "good", "anything", "else", "should", "do", "brings",
            "today", "morning", "great", "thanks", "yes"}
    text = " ".join([
        note.get("subjective", ""),
        " ".join(note.get("objective", []) or []),
        note.get("assessment", ""),
        " ".join(note.get("plan", []) or []),
        " ".join(m.get("name", "") for m in note.get("medications", []) or []),
    ]).lower()
    toks = {t for t in _WORD.findall(text) if t not in stop and len(t) > 2}
    return toks


@dataclass
class CaseScore:
    encounter_id: str
    structural_validity: bool
    phi_leak_clean: bool
    icd_grounded: bool
    code_accuracy: bool
    completeness: float          # recall in [0,1]
    faithfulness: float          # judge score in [0,1]
    attempts: int
    status: str
    leaks: list[str] = field(default_factory=list)


# ---- LLM-as-judge (faithfulness) -------------------------------------------

def judge_faithfulness(transcript: str, note: dict, *, offline: bool = True) -> float:
    """Score how well the note is supported by the transcript, in [0,1].

    OFFLINE judge: fraction of the note's content tokens that actually appear in
    the (redacted) transcript. A deterministic proxy for "is this grounded in
    the source" — unsupported/invented tokens drive the score down.

    PRODUCTION judge (Anthropic): show transcript + note, ask the model to flag
    unsupported claims and return a 0-1 faithfulness score. Sketch:

        client.messages.create(model="claude-sonnet-4-6", ...,
            system="You are a strict clinical auditor. Return JSON "
                   "{faithfulness: float, unsupported: [..]}.")
    """
    if not offline:
        import anthropic  # noqa: F401
        # production judge call omitted for the dependency-free demo
        raise NotImplementedError("wire the Anthropic judge here")

    note_toks = _facts(note)
    if not note_toks:
        return 0.0
    src = set(_WORD.findall(transcript.lower()))
    supported = sum(1 for t in note_toks if t in src)
    return round(supported / len(note_toks), 3)


def judge_agreement(judge_scores: list[float], human_labels: list[float],
                    tol: float = 0.15) -> float:
    """Fraction of cases where the judge agrees with a human label within tol.

    Run on a small hand-labeled subset to validate the judge before trusting it.
    """
    if not human_labels:
        return float("nan")
    agree = sum(1 for j, h in zip(judge_scores, human_labels) if abs(j - h) <= tol)
    return round(agree / len(human_labels), 3)


# ---- Per-case scoring ------------------------------------------------------

def score_case(encounter, agent_result, grounded_codes: set[str]) -> CaseScore:
    note = agent_result.note
    gold = encounter.gold_note

    if note is None:
        # blocked (validation give-up or PHI leak) — everything fails cleanly
        return CaseScore(
            encounter_id=encounter.encounter_id,
            structural_validity=False, phi_leak_clean=(agent_result.status != "phi_leak"),
            icd_grounded=False, code_accuracy=False, completeness=0.0,
            faithfulness=0.0, attempts=agent_result.attempts,
            status=agent_result.status,
        )

    structural = validate_draft(note).ok

    leak = detect_phi_leak(json.dumps(note), encounter.planted_phi)

    emitted = set(note.get("icd_candidates", []))
    grounded = all(c in grounded_codes for c in emitted)
    code_ok = set(gold["icd_candidates"]) == emitted

    gold_facts = _facts(gold)
    got_facts = _facts(note)
    completeness = round(len(gold_facts & got_facts) / len(gold_facts), 3) if gold_facts else 0.0

    faithfulness = judge_faithfulness(encounter.transcript, note)

    return CaseScore(
        encounter_id=encounter.encounter_id,
        structural_validity=structural,
        phi_leak_clean=leak.clean,
        icd_grounded=grounded,
        code_accuracy=code_ok,
        completeness=completeness,
        faithfulness=faithfulness,
        attempts=agent_result.attempts,
        status=agent_result.status,
        leaks=leak.leaks,
    )


@dataclass
class Report:
    cases: list[CaseScore]

    def aggregate(self) -> dict:
        n = len(self.cases)
        if n == 0:
            return {}
        return {
            "n_cases": n,
            "structural_validity_rate": round(sum(c.structural_validity for c in self.cases) / n, 3),
            "phi_leak_free_rate": round(sum(c.phi_leak_clean for c in self.cases) / n, 3),
            "icd_grounded_rate": round(sum(c.icd_grounded for c in self.cases) / n, 3),
            "code_accuracy_rate": round(sum(c.code_accuracy for c in self.cases) / n, 3),
            "mean_completeness": round(sum(c.completeness for c in self.cases) / n, 3),
            "mean_faithfulness": round(sum(c.faithfulness for c in self.cases) / n, 3),
            "mean_attempts": round(sum(c.attempts for c in self.cases) / n, 3),
        }

    def to_json(self) -> str:
        return json.dumps({
            "aggregate": self.aggregate(),
            "cases": [c.__dict__ for c in self.cases],
        }, indent=2)

    def to_markdown(self) -> str:
        agg = self.aggregate()
        lines = ["# Eval Report", "", "## Aggregate", ""]
        for k, v in agg.items():
            lines.append(f"- **{k}**: {v}")
        lines += ["", "## Per-encounter", "",
                  "| id | valid | no-leak | grounded | code✓ | complete | faithful | tries |",
                  "|----|-------|---------|----------|-------|----------|----------|-------|"]
        for c in self.cases:
            lines.append(
                f"| {c.encounter_id} | {'✓' if c.structural_validity else '✗'} "
                f"| {'✓' if c.phi_leak_clean else '✗'} "
                f"| {'✓' if c.icd_grounded else '✗'} "
                f"| {'✓' if c.code_accuracy else '✗'} "
                f"| {c.completeness:.2f} | {c.faithfulness:.2f} | {c.attempts} |"
            )
        return "\n".join(lines)
