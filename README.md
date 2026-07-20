# Clinical SOAP Note Agent

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

A clinical documentation pipeline: given a doctor–patient conversation, produce a
structured note with four sections — **S**ubjective, **O**bjective, **A**ssessment,
**P**lan — and then *measure whether that note is actually any good*.

Generating the note is the easy part. Making it trustworthy is the work, and
that's what most of this repo is about.

---

## The finding this project is really about

I evaluated the pipeline on synthetic data I wrote myself. The scores looked
great. But I had written both the conversations *and* the answer key, so they
proved very little.

So I ran the same pipeline against **ACI-BENCH**, an external ambient-documentation
benchmark I had no hand in creating. Completeness rose to **53.8%** against a
rules-based baseline's 20.3%. But faithfulness **dropped to 63.5%** — which looked
like the model was inventing a third of every note.

It wasn't. I dumped every flagged term with its source sentence and reviewed them
by hand. Almost all were **clinically valid paraphrase**, not invention:

| Transcript says | Note says | Metric verdict |
|---|---|---|
| "stuffy nose" | nasal congestion | flagged as unsupported |
| "a bit of a fever" | low-grade fever | flagged as unsupported |
| "no, no fever" | denies fever | flagged as unsupported |

The lexical metric was penalising the model for **writing like a clinician**.

The proof: run that same metric on the *reference note* — the expert-reviewed
answer key. It uses "nasal congestion" too. **It would fail my own metric.** If the
correct answer can't pass your test, the test is wrong.

And the dangerous part: if I had trusted that 63.5% and tuned the prompt to raise
it, I would have pushed the model toward **copying the transcript verbatim** —
raising the score while making the notes clinically worse.

> **My scorer graded the model. Nothing graded my scorer.**

---

## Architecture

```
transcribe → redact PHI → retrieve → extract → validate → guardrail → note
                                        ↑          |
                                        └── retry ─┘  (max 3, then escalate)
```

| Stage | What it does | Why it exists |
|---|---|---|
| **redact** | strips name, MRN, DOB, phone, email **before** the model call | once raw PHI leaves the machine it's gone; cleaning output is theatre |
| **retrieve** | semantic search over a clinical KB, returns valid ICD codes | the model picks from evidence rather than recalling from memory |
| **extract** | LLM fills a strict Pydantic schema | structure is checkable; prose isn't |
| **validate** | schema check **+** every code must be in what retrieval returned | constrains hallucination (does not eliminate it — only as strong as retrieval) |
| **retry** | feeds the *specific* validation error back into the prompt | attempt 2 knows what broke in attempt 1 |
| **guardrail** | scans the finished note for leaked PHI | defence in depth |

Built on **LangGraph** specifically because validation needs to send work *back* to
extraction — a cycle a linear chain can't express.

---

## Human-in-the-loop: knowing what you don't know

First-pass extraction is always incomplete. The tempting fix — prompting the model
to "be more thorough" — is dangerous: it pushes the model to invent findings that
were never discussed. In clinical documentation a fabricated finding is far worse
than a missing one, because a missing one is visible and a fabricated one isn't.

So instead: **the schema knows what a complete note needs.**

- Every clinical item is its own nullable field, so a gap is literally `None`
- A **deterministic** check finds the empty required fields — no model judgment
- The LLM is used *only* to phrase the clarification question
- The system surfaces **relevant transcript evidence** alongside the question
- The clinician decides

> Deterministic where correctness matters. LLM where fluency matters.

The model is never asked *"what did you miss?"* — that's asking it to know what it
doesn't know, and it will confidently invent an answer.

---

## Groundedness ≠ provenance

When a clinician fills a gap, that content isn't in the transcript. My first
instinct was to make clinician answers count as "supported" so the check would
pass. **That was wrong** — it suppressed exactly the signal a clinical record needs.

The audit tracks two **independent** axes:

| | `grounded_to_transcript` | `provenance` | verdict |
|---|---|---|---|
| model extracted it, it checks out | `True` | transcript | fine |
| clinician supplied it, not in transcript | `False` | clinician | **flag — confirm intent** |
| model asserted it, from no source | `False` | unknown | **hallucination** |

The bottom two rows are the *same* lexical failure and *completely different*
clinical events. One is a doctor exercising judgment; one is a machine inventing
content. **Same "not in the transcript" — opposite safety, because one has an
accountable human behind it and one doesn't.**

Averaging them into a single faithfulness score destroys the only distinction that
matters. So there isn't one.

---

## Validating the judge

Replacing a broken lexical metric with an LLM judge only helps if the judge is
better. Trusting it *because it's an LLM* would repeat the original mistake.

So the judge was scored against a human-labelled set probing specific failure modes
— paraphrase, clinical terminology, negation, negation flips, outright invention:

| scorer | precision | recall | Cohen's κ | accuracy |
|---|---|---|---|---|
| lexical (word overlap) | 0.75 | 0.60 | **0.25** | 0.62 |
| LLM judge | 1.00 | 0.80 | **0.75** | 0.88 |

κ = 0.25 is barely better than chance. κ = 0.75 is substantial agreement.

**The judge is validated, not trusted.** *(One non-clinician labeller, 8 cases — a
signal, not a proof.)*

---

## What's in this repo

```
soap_agent/
  agent.py                  LangGraph pipeline: retrieve → extract → validate → retry
  schema.py                 Pydantic contract for the note
  schema_granular.py        granular nullable fields — makes gaps detectable
  retriever.py              semantic retrieval (sentence-transformers) + score floor
  guardrails.py             PHI redaction (input) + leak detection (output)
  extractor.py              the LLM call
  coverage.py               deterministic gap detection + clarification questions
  evidence_clarification.py surfaces relevant transcript lines for each gap
  api.py                    FastAPI service

evals/
  harness.py                synthetic-gold scoring
  real_data_eval.py         ACI-BENCH evaluation + rules-based baseline
  error_analysis.py         manual review worksheet — how the metric bug was found
  validate_judge.py         grades the grader against human labels

scripts/
  demo.py                   one encounter, full trace
  run_eval.py               synthetic evaluation
  coverage_demo.py          gap detection + clarification loop
  groundedness.py           two-axis provenance audit
  evidence_demo.py          evidence-surfaced clarification

tests/                      27 tests across pipeline, retrieval, and coverage
THREAT_MODEL.md             PHI handling, scope, and what this is NOT
```

---

## Running it

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-...

python scripts/demo.py                 # one note, full trace
python scripts/run_eval.py             # synthetic gold evaluation
python evals/real_data_eval.py --live  # ACI-BENCH + baseline comparison
python evals/error_analysis.py --live  # the manual review that found the bug
python evals/validate_judge.py --live  # grade the grader
python scripts/coverage_demo.py        # gap detection + clarification
python scripts/groundedness.py         # provenance audit
python -m pytest tests/ -v             # 27 tests
```

---

## About the benchmark

**ACI-BENCH** (Yim et al., *Nature Scientific Data*, 2023) is an external
ambient-documentation benchmark. Being precise about what it is, because it matters:

- The encounters are **role-played** by a certified physician and a lay volunteer,
  or **constructed by medical experts**. They are not recordings of real patient
  visits, and contain no patient data.
- The reference notes were **machine-drafted, then corrected and rewritten by
  clinical scribes and physicians** — expert-reviewed, not written from scratch.

What makes it useful is that it's **external** (~1,220 words per encounter versus
my 40-word synthetic ones) and **I authored none of it**.

---

## Honest limitations

- **Transcription and diarization are stubbed.** Reading a transcript is not the
  same as processing audio. Real ambient audio is the hard part: a two-person
  conversation through one microphone can sound like many more depending on
  distance and room acoustics, and a SOAP note fundamentally depends on knowing
  who said what. Wiring in Whisper is a few hours; diarization is the real blocker.
- **Metrics are lexical proxies, not semantic judgments.** The absolute numbers
  understate quality — they're for *comparing* systems on identical data, not for
  claiming clinical accuracy.
- **Grounding constrains hallucination; it does not eliminate it.** The check is
  only as strong as the retrieval. Bad retrieval means a wrong code can be
  "grounded" and pass.
- **Automated evaluation is not clinical validation.** These are technical and
  semantic quality signals. Whether a note is *clinically adequate* requires
  qualified clinical reviewers, and no amount of code closes that gap.
- **n is small, and I am not a clinician.** Enough to compare configurations, not
  to claim production readiness.

---

## Dataset citation

> Yim, W., Fu, Y., Ben Abacha, A., Snider, N., Lin, T., & Yetisgen, M. (2023).
> *ACI-BENCH: a Novel Ambient Clinical Intelligence Dataset for Benchmarking
> Automatic Visit Note Generation.* **Nature Scientific Data.**

## License

MIT
