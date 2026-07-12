# Real-Data Evaluation — ACI-BENCH

Evaluates the clinical SOAP-note pipeline on **real doctor-patient conversations**
instead of synthetic ones, using EvalLens-style metrics and failure taxonomy.

## Why this exists

The original pipeline was validated on synthetic encounters where I authored both
the transcript *and* the gold note. That produces a clean signal but proves little —
the model is being graded against data shaped to suit it. Near-perfect scores on
your own synthetic data are not evidence that a system works.

**ACI-BENCH is the honest test.**

- **Source:** Yim et al., *"ACI-BENCH: a Novel Ambient Clinical Intelligence Dataset
  for Benchmarking Automatic Visit Note Generation"*, **Nature Scientific Data (2023)**
- **Data:** real transcribed clinical visits (avg **~1,220 words** — vs ~40 in my
  synthetic set), paired with **human-written gold notes** from trained annotators
- **Nothing** in it was authored to suit this pipeline

## What it measures

| Metric | Meaning |
|---|---|
| **completeness** | share of gold clinical content the generated note captures |
| **faithfulness** | share of generated content traceable to the conversation |
| **section coverage** | share of the four SOAP sections populated |
| **approval rate** | notes with no flagged failure |
| **failure modes** | EvalLens taxonomy: `missing_info`, `unsupported_claim`, `bad_format`, `thin_section` |

## The baseline is the point

The harness always runs a **rules-based extractive baseline** (no LLM) alongside the
LLM pipeline. This is deliberate: *a generated note that can't beat naive extraction
isn't earning its cost.* Without a baseline, an impressive-looking score is
uninterpretable.

## Results (baseline, n=20 real encounters)

```
completeness    20.3%    <- naive extraction captures only 1/5 of gold content
faithfulness    99.4%    <- trivially high: it only ever copies, never invents
section cover   95.0%
approval rate    0.0%
failure modes   missing_info 20 · bad_format 3 · thin_section 2
```

Two things worth reading carefully here:

**20.3% completeness** is the honest floor. Real clinical conversations are long,
hedged, and full of small talk; pulling structured content out of them is genuinely
hard. This is the bar the LLM has to clear to justify itself.

**99.4% faithfulness is not a good score — it's a warning.** The baseline only ever
copies verbatim, so of course nothing is "unsupported." A high faithfulness number
paired with a terrible completeness number describes a system that says almost
nothing, very accurately. This is exactly the kind of misleading single-metric story
an error taxonomy is meant to break apart.

## Run it

```bash
# baseline only — no API key required
python evals/real_data_eval.py

# baseline + live LLM comparison — requires ANTHROPIC_API_KEY
python evals/real_data_eval.py --live

# larger split (40 encounters)
python evals/real_data_eval.py --live --data data/aci_bench/aci_bench_test1.csv --n 40
```

## Honest limitations

**The metrics are lexical proxies, not semantic judgments.** Completeness and
faithfulness are computed on content-term overlap. A correct clinical *paraphrase*
("SOB on exertion" for "short of breath when he walks") is scored as a miss. So the
absolute numbers understate true quality — they are useful for **comparing** systems
on identical data, not as a claim about clinical accuracy.

**Section mapping is approximate.** ACI-BENCH notes use clinical headers (CHIEF
COMPLAINT, HPI, PHYSICAL EXAM, ...) which I map onto SOAP. `ASSESSMENT AND PLAN`
sections are counted toward both, which is a judgment call, not ground truth.

**A real evaluation would need clinician review.** Automated proxies catch gross
failures — invented findings, missing sections, empty output. They cannot judge
whether a note is *clinically adequate*. That requires a clinician, which is why
human-in-the-loop review is part of the design and not an afterthought.

**n is small** (20 valid / 40 test). Enough to compare configurations, not enough
to make confident claims about production readiness.
