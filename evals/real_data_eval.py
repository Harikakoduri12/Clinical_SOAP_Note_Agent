"""
Real-Data Evaluation: ACI-BENCH
================================

Runs the clinical SOAP-note pipeline over REAL doctor-patient conversations from
ACI-BENCH (Yim et al., Nature Scientific Data 2023) — a public benchmark for
ambient clinical intelligence — and evaluates the generated notes with EvalLens.

Why this matters
----------------
The original pipeline was validated on synthetic encounters where I authored both
the transcript and the gold note. That guarantees a clean signal but proves little:
the model is graded against data shaped for it.

ACI-BENCH is the honest test. The conversations are real (transcribed clinical
visits, ~1,200 words, full of hedging, interruptions, and small talk) and the gold
notes are human-written by trained annotators. Nothing was authored to suit the
pipeline.

Expect scores to be LOWER than on synthetic data. That drop is the finding, not a
failure — it quantifies the gap between a controlled prototype and real clinical
input, which is exactly what an evaluation harness exists to surface.

Mapping
-------
ACI-BENCH notes use clinical section headers rather than a SOAP schema, so they
are mapped to SOAP before scoring:

    CHIEF COMPLAINT / HPI / REVIEW OF SYSTEMS / SOCIAL+FAMILY+MEDICAL HISTORY
        -> Subjective   (what the patient reports)
    PHYSICAL EXAM / VITALS / RESULTS
        -> Objective    (what was measured or observed)
    ASSESSMENT (incl. ASSESSMENT AND PLAN)
        -> Assessment   (clinical judgment)
    PLAN / INSTRUCTIONS
        -> Plan         (next steps)
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from collections import Counter

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------- section map

SUBJECTIVE = ["CHIEF COMPLAINT", "HISTORY OF PRESENT ILLNESS", "REVIEW OF SYSTEMS",
              "SOCIAL HISTORY", "FAMILY HISTORY", "MEDICAL HISTORY", "PAST HISTORY",
              "MEDICATIONS", "SURGICAL HISTORY", "CURRENT MEDICATIONS",
              "PAST MEDICAL HISTORY", "ALLERGIES"]
OBJECTIVE = ["PHYSICAL EXAM", "VITALS", "RESULTS", "PHYSICAL EXAMINATION",
             "VITAL SIGNS", "LABORATORY RESULTS", "EXAM"]
ASSESSMENT = ["ASSESSMENT", "IMPRESSION", "ASSESSMENT AND PLAN", "DIAGNOSIS"]
PLAN = ["PLAN", "INSTRUCTIONS", "TREATMENT PLAN", "FOLLOW-UP", "DISPOSITION"]

_HEADER = re.compile(r"^([A-Z][A-Z /&'\-\(\)0-9]{2,})\s*$", re.M)


def parse_note_sections(note: str) -> dict[str, str]:
    """Split a clinical note into {HEADER: body} using its ALL-CAPS headers."""
    note = str(note)
    hits = list(_HEADER.finditer(note))
    out: dict[str, str] = {}
    for i, m in enumerate(hits):
        head = m.group(1).strip()
        start = m.end()
        end = hits[i + 1].start() if i + 1 < len(hits) else len(note)
        body = note[start:end].strip()
        if body:
            out[head] = body
    return out


def to_soap(note: str) -> dict[str, str]:
    """Map a real ACI-BENCH note onto the four SOAP fields."""
    secs = parse_note_sections(note)
    buckets = {"subjective": [], "objective": [], "assessment": [], "plan": []}

    for head, body in secs.items():
        h = head.upper()
        if any(k in h for k in ASSESSMENT) and "PLAN" in h:
            # "ASSESSMENT AND PLAN" contributes to both
            buckets["assessment"].append(body)
            buckets["plan"].append(body)
        elif any(h.startswith(k) or k in h for k in ASSESSMENT):
            buckets["assessment"].append(body)
        elif any(h.startswith(k) or k in h for k in PLAN):
            buckets["plan"].append(body)
        elif any(h.startswith(k) or k in h for k in OBJECTIVE):
            buckets["objective"].append(body)
        elif any(h.startswith(k) or k in h for k in SUBJECTIVE):
            buckets["subjective"].append(body)
        else:
            # unmapped header: default to subjective (narrative history)
            buckets["subjective"].append(body)

    return {k: "\n".join(v).strip() for k, v in buckets.items()}


# ------------------------------------------------------------------- scoring

_WORD = re.compile(r"[a-z0-9]+")
_STOP = {
    "the","and","for","with","that","this","was","were","has","have","had","not",
    "you","your","are","but","his","her","she","him","they","them","their","from",
    "who","which","will","would","can","could","should","been","being","also",
    "any","all","its","it's","there","here","then","than","some","did","does",
    "doing","done","get","got","just","like","about","into","over","out","off",
    "patient","reports","report","states","note","noted","today","doctor","okay",
    "yeah","right","well","know","think","going","little","bit","lot","really",
    "mm-hmm","um","uh","so","he","of","to","a","an","in","on","is","at","as","i",
    "we","it","or","if","be","do","my","me","no","yes","ok",
}


def content_terms(text: str) -> set[str]:
    """Clinical content terms: drop stopwords and conversational filler."""
    return {t for t in _WORD.findall(str(text).lower())
            if t not in _STOP and len(t) > 2}


def completeness(generated: dict, gold: dict) -> float:
    """Recall of gold clinical content captured by the generated note."""
    g = content_terms(" ".join(gold.values()))
    if not g:
        return 0.0
    return len(g & content_terms(" ".join(generated.values()))) / len(g)


def faithfulness(generated: dict, transcript: str) -> float:
    """Fraction of generated content that is traceable to the conversation.

    A proxy, not a semantic judgment: it catches content invented wholesale, but
    a paraphrase of something truly said may be scored as unsupported. Documented
    as a limitation rather than papered over.
    """
    gen = content_terms(" ".join(generated.values()))
    if not gen:
        return 0.0
    src = set(_WORD.findall(str(transcript).lower()))
    return len([t for t in gen if t in src]) / len(gen)


def section_coverage(generated: dict) -> float:
    """Structural validity: fraction of the four SOAP sections populated."""
    return sum(1 for v in generated.values() if str(v).strip()) / 4.0


# --------------------------------------------------- EvalLens error taxonomy

MISSING_INFO = "missing_info"
UNSUPPORTED = "unsupported_claim"
BAD_FORMAT = "bad_format"
THIN_SECTION = "thin_section"


def classify_failures(generated: dict, gold: dict, transcript: str,
                      comp: float, faith: float) -> list[str]:
    """EvalLens-style failure taxonomy, applied to real notes."""
    errs = []
    if comp < 0.55:
        errs.append(MISSING_INFO)
    if faith < 0.80:
        errs.append(UNSUPPORTED)
    if section_coverage(generated) < 1.0:
        errs.append(BAD_FORMAT)
    for name, body in generated.items():
        if body and len(content_terms(body)) < 4:
            errs.append(THIN_SECTION)
            break
    return errs


# ------------------------------------------------------------------ pipeline

def generate_note(transcript: str, *, offline: bool) -> dict:
    """Produce a SOAP note from the conversation.

    offline=True  -> deterministic extractive baseline (no API key needed)
    offline=False -> live Anthropic API (the real pipeline)
    """
    if not offline:
        return _generate_live(transcript)
    return _generate_baseline(transcript)


def _generate_baseline(transcript: str) -> dict:
    """A rules-based extractive baseline.

    This exists so the harness runs without an API key, and — more usefully — so
    the LLM has something to be compared AGAINST. A generated note that cannot
    beat naive extraction is not earning its cost.
    """
    doctor, patient = [], []
    for line in str(transcript).splitlines():
        line = line.strip()
        if line.startswith("[patient]"):
            patient.append(line.replace("[patient]", "").strip())
        elif line.startswith("[doctor]"):
            doctor.append(line.replace("[doctor]", "").strip())

    d_text = " ".join(doctor)
    p_text = " ".join(patient)

    exam_cues = ("exam", "listen", "heart", "lungs", "blood pressure", "vitals",
                 "temperature", "murmur", "swelling", "tender", "palpation")
    plan_cues = ("we'll", "we will", "i want you to", "let's", "continue",
                 "start", "order", "refer", "follow up", "prescribe", "schedule")
    assess_cues = ("i think", "diagnosis", "impression", "appears", "consistent with",
                   "looks like", "assessment")

    def pick(sentences, cues, limit):
        out = []
        for s in sentences:
            low = s.lower()
            if any(c in low for c in cues):
                out.append(s)
            if len(out) >= limit:
                break
        return " ".join(out)

    d_sents = re.split(r"(?<=[.?!])\s+", d_text)

    return {
        "subjective": p_text[:900],
        "objective": pick(d_sents, exam_cues, 6)[:600],
        "assessment": pick(d_sents, assess_cues, 3)[:400],
        "plan": pick(d_sents, plan_cues, 6)[:600],
    }


def _generate_live(transcript: str) -> dict:
    """Live LLM generation via the Anthropic API."""
    import json
    import anthropic

    client = anthropic.Anthropic()
    prompt = (
        "You are a clinical documentation assistant. Read the doctor-patient "
        "conversation and write a SOAP note.\n\n"
        "Rules:\n"
        "- Use ONLY information stated in the conversation. Do not infer or invent.\n"
        "- subjective: what the patient reports (symptoms, history, complaints)\n"
        "- objective: measured or observed findings (vitals, exam, results)\n"
        "- assessment: the clinician's judgment/diagnosis\n"
        "- plan: next steps (medications, tests, referrals, follow-up)\n\n"
        "Return ONLY a JSON object with keys: subjective, objective, assessment, plan. "
        "Each value is a string. No markdown, no preamble.\n\n"
        f"CONVERSATION:\n{transcript}"
    )
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {"subjective": "", "objective": "", "assessment": "", "plan": ""}
    return {k: str(data.get(k, "")) for k in
            ("subjective", "objective", "assessment", "plan")}


# ---------------------------------------------------------------------- main

def evaluate(df: pd.DataFrame, *, offline: bool, label: str) -> dict:
    rows = []
    errors: Counter = Counter()
    start = time.perf_counter()

    for _, r in df.iterrows():
        transcript = r["dialogue"]
        gold = to_soap(r["note"])
        gen = generate_note(transcript, offline=offline)

        comp = completeness(gen, gold)
        faith = faithfulness(gen, transcript)
        cov = section_coverage(gen)
        errs = classify_failures(gen, gold, transcript, comp, faith)
        for e in errs:
            errors[e] += 1

        rows.append({
            "encounter_id": r["encounter_id"],
            "completeness": comp,
            "faithfulness": faith,
            "coverage": cov,
            "errors": errs,
            "clean": len(errs) == 0,
        })

    elapsed = time.perf_counter() - start
    n = len(rows)
    return {
        "label": label,
        "n": n,
        "completeness": sum(r["completeness"] for r in rows) / n,
        "faithfulness": sum(r["faithfulness"] for r in rows) / n,
        "coverage": sum(r["coverage"] for r in rows) / n,
        "approval_rate": sum(1 for r in rows if r["clean"]) / n,
        "errors": dict(errors.most_common()),
        "latency_s": elapsed,
        "rows": rows,
    }


def report(res: dict) -> None:
    print(f"\n{'=' * 66}")
    print(f"  {res['label']}   (n={res['n']} real encounters)")
    print("=" * 66)
    print(f"  completeness    {res['completeness']:.1%}   "
          f"(gold clinical content captured)")
    print(f"  faithfulness    {res['faithfulness']:.1%}   "
          f"(content traceable to the conversation)")
    print(f"  section cover   {res['coverage']:.1%}   "
          f"(SOAP sections populated)")
    print(f"  approval rate   {res['approval_rate']:.1%}   "
          f"(notes with no flagged failure)")
    print(f"  latency         {res['latency_s']:.1f}s total")
    print()
    if res["errors"]:
        print("  Failure modes (EvalLens taxonomy):")
        for k, v in res["errors"].items():
            print(f"    {k:<20} {v}")
    else:
        print("  No failures flagged.")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/tmp/acib/data/challenge_data/valid.csv")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--live", action="store_true",
                    help="use the live Anthropic API (needs ANTHROPIC_API_KEY)")
    args = ap.parse_args()

    df = pd.read_csv(args.data).head(args.n)

    print()
    print("REAL-DATA EVALUATION — ACI-BENCH")
    print("Yim et al., Nature Scientific Data (2023)")
    print(f"Real doctor-patient conversations, human-written gold notes.")
    print(f"Avg dialogue length: {int(df['dialogue'].str.split().str.len().mean())} words")

    base = evaluate(df, offline=True, label="Extractive baseline (no LLM)")
    report(base)

    if args.live:
        live = evaluate(df, offline=False, label="LLM pipeline (live Anthropic API)")
        report(live)

        print("=" * 66)
        print("  COMPARISON")
        print("=" * 66)
        row = "  {:<16}{:>14}{:>14}{:>12}"
        print(row.format("metric", "baseline", "LLM", "delta"))
        print("  " + "-" * 56)
        for k in ("completeness", "faithfulness", "approval_rate"):
            b, l = base[k], live[k]
            print(row.format(k, f"{b:.1%}", f"{l:.1%}", f"{(l-b)*100:+.0f} pp"))
        print()
        gain = live["completeness"] - base["completeness"]
        if gain > 0.05:
            print(f"  -> The LLM pipeline beats naive extraction by "
                  f"{gain*100:.0f} points of completeness, so it is earning its cost.")
        else:
            print("  -> The LLM does NOT clearly beat naive extraction here. "
                  "That is a real finding, not a bug to hide.")
        print()


if __name__ == "__main__":
    main()
