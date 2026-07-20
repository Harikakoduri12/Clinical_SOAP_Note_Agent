"""
Error Analysis: why did faithfulness drop?
===========================================

The real-data run showed the LLM roughly doubling completeness (20% -> 54%) while
faithfulness FELL (99% -> 64%). Before concluding "the model hallucinates," the
honest move is to look at what the metric actually flagged.

There are two very different explanations, and the automated score cannot tell them
apart:

  PARAPHRASE   The model wrote "malaise" for "feeling out of sorts", or "dyspnea on
               exertion" for "short of breath when I walk". Clinically correct.
               The metric is wrong, not the model.

  INVENTION    The model wrote a finding, value, or plan item that was never
               discussed. The model is wrong. This is the real risk in clinical
               documentation.

This script dumps every flagged term with its surrounding sentence so a human can
classify it. That human is you. The output is a worksheet, not a verdict.

Usage:
    python evals/error_analysis.py --live -n 3      # 3 encounters, live LLM
    python evals/error_analysis.py -n 3             # 3 encounters, baseline
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from real_data_eval import (to_soap, content_terms, completeness, faithfulness,
                            generate_note, _WORD)


def sentence_for(term: str, note_section: str) -> str:
    """Find the sentence in the generated note where a flagged term appears."""
    for sent in re.split(r"(?<=[.;!?])\s+", note_section):
        if term in sent.lower():
            return sent.strip()
    return ""


def analyze(transcript: str, generated: dict, gold: dict, eid: str) -> None:
    src_terms = set(_WORD.findall(str(transcript).lower()))
    gen_terms = content_terms(" ".join(generated.values()))

    unsupported = sorted(t for t in gen_terms if t not in src_terms)
    gold_terms = content_terms(" ".join(gold.values()))
    missed = sorted(t for t in gold_terms if t not in gen_terms)

    comp = completeness(generated, gold)
    faith = faithfulness(generated, transcript)

    print("\n" + "=" * 74)
    print(f"  ENCOUNTER {eid}")
    print(f"  completeness {comp:.0%}   faithfulness {faith:.0%}")
    print("=" * 74)

    # ---- the flagged content, with the sentence it came from ----
    print(f"\n  [A] FLAGGED AS UNSUPPORTED  ({len(unsupported)} terms)")
    print("  " + "-" * 70)
    print("  For each: is this PARAPHRASE (metric wrong) or INVENTION (model wrong)?")
    print()

    shown = 0
    for term in unsupported:
        # find which section, and the sentence
        sent, sec_name = "", ""
        for name, body in generated.items():
            s = sentence_for(term, str(body))
            if s:
                sent, sec_name = s, name
                break
        if not sent:
            continue
        shown += 1
        if shown > 18:
            print(f"    ... and {len(unsupported) - 18} more")
            break
        print(f"    * '{term}'  [{sec_name}]")
        print(f"      \"{sent[:150]}\"")
        print(f"      -> PARAPHRASE / INVENTION ? ____________")
        print()

    # ---- what got missed ----
    print(f"\n  [B] GOLD CONTENT MISSED  ({len(missed)} terms)")
    print("  " + "-" * 70)
    print("    " + ", ".join(missed[:35]))
    if len(missed) > 35:
        print(f"    ... and {len(missed) - 35} more")

    # ---- the note itself, so you can read it ----
    print(f"\n  [C] GENERATED NOTE")
    print("  " + "-" * 70)
    for name, body in generated.items():
        body = str(body).strip()
        print(f"    {name.upper()}:")
        if body:
            for line in _wrap(body, 66):
                print(f"      {line}")
        else:
            print("      (empty)")
        print()

    print(f"  [D] GOLD NOTE (human-written)")
    print("  " + "-" * 70)
    for name, body in gold.items():
        body = str(body).strip()
        print(f"    {name.upper()}:")
        if body:
            for line in _wrap(body, 66)[:6]:
                print(f"      {line}")
            if len(_wrap(body, 66)) > 6:
                print("      ...")
        else:
            print("      (empty)")
        print()


def _wrap(text: str, width: int) -> list[str]:
    words, lines, cur = str(text).split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    default_data = os.path.normpath(
        os.path.join(here, "..", "data", "aci_bench", "aci_bench_valid.csv"))

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=default_data)
    ap.add_argument("-n", type=int, default=3, help="how many encounters to inspect")
    ap.add_argument("--live", action="store_true", help="use the live Anthropic API")
    args = ap.parse_args()

    df = pd.read_csv(args.data).head(args.n)

    print()
    print("#" * 74)
    print("  ERROR ANALYSIS — why did faithfulness drop?")
    print(f"  {'LIVE LLM' if args.live else 'BASELINE (no LLM)'}   n={len(df)}")
    print("#" * 74)
    print("""
  Read section [A] for each encounter. For every flagged term, decide:

    PARAPHRASE  the model said the same clinical thing in different words
                (correct content, metric limitation)

    INVENTION   the model asserted something never discussed
                (real hallucination — the actual clinical risk)

  Tally them. The ratio is your finding.
""")

    for _, r in df.iterrows():
        gold = to_soap(r["note"])
        gen = generate_note(r["dialogue"], offline=not args.live)
        analyze(r["dialogue"], gen, gold, r["encounter_id"])

    print("\n" + "=" * 74)
    print("  TALLY  (fill this in yourself)")
    print("=" * 74)
    print("""
    Paraphrase (metric wrong) : ____
    Invention  (model wrong)  : ____

    If mostly PARAPHRASE  -> the 64% faithfulness understates the model; the fix
                             is a better metric (semantic similarity / LLM judge),
                             not a better prompt.

    If mostly INVENTION   -> the model really is asserting unsupported content;
                             the fix is prompt constraints, grounding, and a
                             guardrail that blocks unsupported claims.

    Either way: this is the error analysis that turns a number into a decision.
""")


if __name__ == "__main__":
    main()
