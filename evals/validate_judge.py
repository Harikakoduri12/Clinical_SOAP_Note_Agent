#!/usr/bin/env python3
"""
Validate the JUDGE — because nothing was grading the grader.

THE WHOLE PROJECT IN ONE PROBLEM
--------------------------------
The core finding was: my scorer graded the model, and nothing graded my scorer.
The lexical faithfulness metric was wrong — it flagged clinical paraphrase as
hallucination — and I only found out by reading outputs by hand.

The fix was to introduce an LLM judge (groundedness.py --judge). But an LLM judge
is JUST ANOTHER SCORER. Trusting it because it's an LLM would repeat the exact
mistake: swapping one unvalidated grader for another and believing it.

So before the judge is allowed to grade anything, it has to earn that right by
agreeing with human judgment on cases where the correct answer is known.

    the judge grades the note.
    THIS grades the judge.

HOW
---
A small labelled set where each item has a human-assigned ground truth:

    SUPPORTED     the transcript supports this (including as paraphrase)
    UNSUPPORTED   the transcript does not support this

Then run BOTH scorers over it — the lexical one and the LLM judge — and measure
each against the human labels with precision, recall, and Cohen's kappa (agreement
corrected for chance).

The expected result, and the point of the exercise:

    lexical scorer  -> high recall, terrible precision. It flags paraphrase as
                       unsupported, so it "catches" everything and is right by
                       accident. Kappa near zero.
    LLM judge       -> should agree with the human far more often, especially on
                       paraphrase and negation. IF it does, it has earned the right
                       to be the grader. IF it doesn't, it hasn't, and I keep
                       looking.

HONEST LIMITS
-------------
- One labeller, not a clinician. No inter-rater agreement. Small n. This is a
  signal about the judge, not a certification of it.
- An LLM judge shares blind spots with the model it grades — they can be wrong in
  the same way. Human labels are the only thing outside that loop, which is exactly
  why they are the ground truth here.

Run:
    python evals/validate_judge.py            # lexical vs offline stand-in
    python evals/validate_judge.py --live     # lexical vs the real LLM judge
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from scripts.groundedness import grounded_lexical, grounded_judge     # noqa: E402


# --------------------------------------------------- the human-labelled set

@dataclass
class Case:
    claim: str
    transcript: str
    human_label: str      # SUPPORTED or UNSUPPORTED — the ground truth
    tests: str            # what failure mode this case probes


SUPPORTED = "SUPPORTED"
UNSUPPORTED = "UNSUPPORTED"

# Each case is chosen to probe a specific failure. The paraphrase and negation
# cases are where lexical scoring is known to break — those are the ones that
# separate a real judge from a word-matcher.
CASES = [
    Case("Sore throat for three days.",
         "Patient: I've had a sore throat for about three days.",
         SUPPORTED, "literal match — both scorers should get this"),

    Case("Nasal congestion.",
         "Patient: my nose is really stuffy.",
         SUPPORTED,
         "PARAPHRASE — 'nasal congestion' vs 'stuffy nose'. Lexical FAILS this: "
         "no shared words, so it calls a correct paraphrase unsupported."),

    Case("Patient denies fever.",
         "Provider: any fever? Patient: no, no fever.",
         SUPPORTED,
         "CLINICAL NEGATION PHRASING — 'denies fever' is the correct way to record "
         "'no fever'. Lexical sees 'denies' isn't in the transcript and FAILS it."),

    Case("Bilateral basilar crackles.",
         "Provider: I can hear some crackling at the bases of both lungs.",
         SUPPORTED,
         "CLINICAL TERM for a plain-language finding. Lexical FAILS — no overlap."),

    Case("Blood pressure 128/82.",
         "Provider: let's check your throat. Looks red.",
         UNSUPPORTED,
         "GENUINE INVENTION — no BP was taken. Both scorers SHOULD flag this."),

    Case("Patient reports chest pain.",
         "Provider: any chest pain? Patient: no, none at all.",
         UNSUPPORTED,
         "NEGATION FLIP — the transcript says the OPPOSITE. 'chest' and 'pain' are "
         "both present, so lexical calls it SUPPORTED. The judge must catch the flip."),

    Case("History of diabetes.",
         "Patient: I've never had any blood sugar problems.",
         UNSUPPORTED,
         "CONTRADICTION — transcript denies it. Lexical may match on 'diabetes'-"
         "adjacent terms; the judge must see the denial."),

    Case("Prescribed acetaminophen.",
         "Provider: take some acetaminophen for the fever.",
         SUPPORTED, "literal match — control case"),
]


# ------------------------------------------------------------ scoring

@dataclass
class Score:
    name: str
    tp: int = 0    # correctly said SUPPORTED
    tn: int = 0    # correctly said UNSUPPORTED
    fp: int = 0    # said SUPPORTED when human said UNSUPPORTED
    fn: int = 0    # said UNSUPPORTED when human said SUPPORTED (the paraphrase bug)

    @property
    def precision(self):
        d = self.tp + self.fp
        return self.tp / d if d else 0.0

    @property
    def recall(self):
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    @property
    def accuracy(self):
        n = self.tp + self.tn + self.fp + self.fn
        return (self.tp + self.tn) / n if n else 0.0

    @property
    def kappa(self):
        n = self.tp + self.tn + self.fp + self.fn
        if not n:
            return 0.0
        po = (self.tp + self.tn) / n
        p_yes_h = (self.tp + self.fn) / n
        p_yes_s = (self.tp + self.fp) / n
        pe = p_yes_h * p_yes_s + (1 - p_yes_h) * (1 - p_yes_s)
        return (po - pe) / (1 - pe) if pe != 1 else 0.0


def score_scorer(name: str, verdict_fn, threshold: float = 0.5) -> Score:
    out = Score(name=name)
    for c in CASES:
        said_supported = verdict_fn(c.claim, c.transcript) >= threshold
        human_supported = (c.human_label == SUPPORTED)
        if said_supported and human_supported:
            out.tp += 1
        elif not said_supported and not human_supported:
            out.tn += 1
        elif said_supported and not human_supported:
            out.fp += 1
        else:
            out.fn += 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true",
                    help="use the real LLM judge (needs ANTHROPIC_API_KEY)")
    args = ap.parse_args()

    print("\n" + "=" * 78)
    print("  VALIDATING THE JUDGE against human labels")
    print("=" * 78)
    print(f"  cases: {len(CASES)}   judge: "
          f"{'LIVE LLM' if args.live else 'offline stand-in'}")
    print()

    lex = score_scorer("lexical (word overlap)", grounded_lexical)
    judge = score_scorer(
        "LLM judge",
        lambda c, t: grounded_judge(c, t) if args.live
        else _offline_judge(c, t))

    row = "  {:<26}{:>10}{:>10}{:>10}{:>10}"
    print(row.format("scorer", "precis.", "recall", "kappa", "accuracy"))
    print("  " + "-" * 66)
    for s in (lex, judge):
        print(row.format(s.name, f"{s.precision:.2f}", f"{s.recall:.2f}",
                         f"{s.kappa:.2f}", f"{s.accuracy:.2f}"))
    print()

    # per-case detail so the failures are visible, not just summarised
    print("  Where the lexical scorer breaks (human label vs lexical verdict):")
    print("  " + "-" * 66)
    for c in CASES:
        lex_says = SUPPORTED if grounded_lexical(c.claim, c.transcript) >= 0.5 \
            else UNSUPPORTED
        mark = "ok " if lex_says == c.human_label else "XX "
        print(f"    {mark} human={c.human_label:<12} lexical={lex_says:<12} "
              f"\"{c.claim[:36]}\"")
    print()

    print("  " + "-" * 66)
    print(f"  lexical kappa   = {lex.kappa:.2f}")
    print(f"  judge kappa     = {judge.kappa:.2f}")
    print()
    if judge.kappa > lex.kappa:
        print("  The judge agrees with human labels MORE than the lexical scorer.")
        print("  On this set, it has earned the right to be the grader. Note the")
        print("  limits: one non-clinician labeller, small n — a signal, not proof.")
    else:
        print("  The judge did NOT beat the lexical scorer here. It has NOT earned")
        print("  the right to grade. That is a real result — do not deploy a judge")
        print("  that can't outperform word-matching against human judgment.")
    print()
    print("  Either way, the point stands: the judge is now VALIDATED rather than")
    print("  TRUSTED. Nothing grades the grader unless you build the thing that does.")
    print()


def _offline_judge(claim: str, transcript: str) -> float:
    """Negation-aware stand-in so the suite runs without an API key.

    Weak on purpose — it demonstrates the SHAPE of a semantic judge and will miss
    things the real judge catches (paraphrase especially). Labelled as a stand-in,
    not presented as the real thing.
    """
    import re
    t = transcript.lower()
    terms = [w for w in re.findall(r"[a-z]+", claim.lower()) if len(w) > 3]
    if not terms:
        return 1.0
    for term in terms:
        for m in re.finditer(re.escape(term), t):
            window = t[max(0, m.start() - 35):m.start()]
            if re.search(r"\b(no|not|never|denies|denied|none|without)\b", window):
                return 0.0     # negated -> unsupported
    present = sum(1 for w in terms if w in t) / len(terms)
    return 1.0 if present >= 0.5 else 0.0


if __name__ == "__main__":
    main()
