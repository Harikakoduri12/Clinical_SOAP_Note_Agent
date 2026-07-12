# Clinical SOAP Note Agent — Evaluation

[![eval](https://github.com/Harikakoduri12/Clinical_SOAP_Note_Agent/actions/workflows/eval.yml/badge.svg)](https://github.com/Harikakoduri12/Clinical_SOAP_Note_Agent/actions/workflows/eval.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

Evaluation suite for a clinical **SOAP-note generation** pipeline: given a
doctor–patient conversation, produce a structured note with four sections —
**S**ubjective, **O**bjective, **A**ssessment, **P**lan — and then *measure how
good that note actually is*.

The interesting part here is not the generation, it's the **grading**. Two
evaluation tracks are included, built on one principle: an impressive-looking
score on data you shaped yourself proves almost nothing. So every metric is
paired with either a baseline to beat or a real-world dataset that was never
authored to suit the model.

---

## What's in this repo

```
data/aci_bench/          Real doctor-patient conversations + human-written gold notes
  aci_bench_valid.csv      2,080 rows  (validation split)
  aci_bench_test1.csv      4,047 rows  (test split)
evals/
  real_data_eval.py      Real-data evaluation on ACI-BENCH  (runs standalone)
  harness.py             Deterministic + LLM-as-judge scoring harness
  REAL_DATA_README.md    Deep-dive write-up of the real-data track
  report.json / .md      Saved results from the synthetic harness run
```

> **Scope note.** This repo is the **evaluation layer**. `real_data_eval.py` is
> fully self-contained and runs on its own. `harness.py` imports the generation
> pipeline itself (`soap_agent.schema`, `soap_agent.guardrails`), which lives in
> the parent project and is **not** included here — the saved `report.json` /
> `report.md` are the output of running it against that pipeline.

---

## Two evaluation tracks

### 1. Real-data evaluation — ACI-BENCH  (`evals/real_data_eval.py`)

The honest test. Instead of synthetic encounters, it runs on **ACI-BENCH**
(Yim et al., *Nature Scientific Data*, 2023): real transcribed clinical visits
(avg ~1,220 words, full of hedging, interruptions, and small talk) paired with
**human-written gold notes** from trained annotators. Nothing in it was shaped
to suit the pipeline.

It always runs a **rules-based extractive baseline (no LLM)** alongside the LLM
pipeline — because *a generated note that can't beat naive extraction isn't
earning its cost.*

| Metric | Meaning |
|---|---|
| **completeness** | share of gold clinical content the note captures (recall) |
| **faithfulness** | share of generated content traceable to the conversation |
| **section coverage** | share of the four SOAP sections populated |
| **approval rate** | notes with no flagged failure |
| **failure modes** | EvalLens taxonomy: `missing_info`, `unsupported_claim`, `bad_format`, `thin_section` |

Baseline results (n=20 real encounters):

```
completeness    20.3%    <- naive extraction captures only 1/5 of gold content
faithfulness    99.4%    <- trivially high: it only ever copies, never invents
section cover   95.0%
approval rate    0.0%
```

That 99.4% faithfulness is **a warning, not a win** — a system that says almost
nothing, very accurately, still scores high on faithfulness alone. Pairing it
with the 20.3% completeness floor is what makes the number honest. See
[`evals/REAL_DATA_README.md`](evals/REAL_DATA_README.md) for the full analysis.

### 2. Synthetic harness — deterministic + LLM-as-judge  (`evals/harness.py`)

A stricter, structured scorer for the generation pipeline. It splits checks into
two tiers on purpose:

- **Deterministic (non-gameable, no AI):** schema/structural validity, **PHI
  leakage** (must be 0), ICD-code grounding, code accuracy, and completeness
  (gold-fact recall).
- **LLM-as-judge (fuzzy part only):** faithfulness — with a `judge_agreement`
  step to spot-check the judge against human labels before trusting it.
  *Distrusting your own grader is the point.*

Saved results ([`evals/report.md`](evals/report.md), n=5): structural validity,
PHI-leak-free, ICD-grounding, and code accuracy all **1.0**; mean completeness
**0.636**; mean faithfulness **1.0**.

---

## Running it

Requires Python 3.10+ and `pandas` (plus `anthropic` for the `--live` path).

```bash
pip install pandas anthropic

# Baseline only — no API key required
python evals/real_data_eval.py --data data/aci_bench/aci_bench_valid.csv

# Baseline + live LLM comparison — needs ANTHROPIC_API_KEY
export ANTHROPIC_API_KEY=sk-...
python evals/real_data_eval.py --data data/aci_bench/aci_bench_valid.csv --live

# Larger split (40 encounters from the test set)
python evals/real_data_eval.py --data data/aci_bench/aci_bench_test1.csv --n 40 --live
```

The `--live` run prints a side-by-side **baseline vs. LLM** comparison and states
plainly whether the LLM clears the baseline — if it doesn't, that's reported as a
finding, not hidden.

---

## Honest limitations

- **Metrics are lexical proxies, not semantic judgments.** Completeness and
  faithfulness are computed on content-term overlap, so a correct clinical
  *paraphrase* is scored as a miss. The absolute numbers understate true quality;
  they're for **comparing** systems on identical data, not for claiming clinical
  accuracy.
- **Section mapping is approximate.** ACI-BENCH clinical headers (CHIEF
  COMPLAINT, HPI, PHYSICAL EXAM, …) are mapped onto SOAP, and `ASSESSMENT AND
  PLAN` is counted toward both — a judgment call, not ground truth.
- **A real evaluation needs clinician review.** Automated proxies catch gross
  failures (invented findings, missing sections, empty output); they cannot judge
  clinical adequacy. Human-in-the-loop review is part of the design, not an
  afterthought.
- **n is small** — enough to compare configurations, not to claim production
  readiness.

---

## Dataset citation

> Yim, W., Fu, Y., Ben Abacha, A., Snider, N., Lin, T., & Yetisgen, M. (2023).
> *ACI-BENCH: a Novel Ambient Clinical Intelligence Dataset for Benchmarking
> Automatic Visit Note Generation.* **Nature Scientific Data.**
