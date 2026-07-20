# Threat Model & Responsible-AI Notes

## Data

This project processes **synthetic data only**. Every patient identity,
conversation, and identifier is generated (Faker + templated scenarios). **No
real PHI is processed anywhere in this codebase.** The synthetic data exists so
the pipeline can be built and evaluated without touching protected information.

## Guardrails and what they guarantee

There are two guardrails, at two points:

1. **Input redaction (pre-LLM)** — `guardrails.redact_transcript` removes
   identifiers (names, MRN, DOB, phone, email) before the transcript reaches the
   model. This is defense-in-depth: the model should never need raw identifiers
   to write a clinical note.

2. **Output leak detection (post-generation)** — `guardrails.detect_phi_leak`
   scans the final note for any *known* injected identifier. Because the
   synthetic generator records exactly what PHI it planted, leak detection recall
   is measured exactly, and any leak is a hard failure that blocks the output.

**What this does NOT guarantee.** The offline redactor is regex-based. Regexes
miss free-text names and unusual formats; that is why the input guardrail here is
a *mitigation*, not a compliance control. A production system must use a clinical
NER de-identifier (e.g. Microsoft Presidio or a spaCy clinical model) and should
still treat de-identification as best-effort rather than perfect.

## Grounding (anti-hallucination)

ICD-10 codes in the output are constrained to codes present in the chunks
retrieved for that encounter (`agent.node_validate`). Ungrounded codes fail
validation. This turns "we hope it didn't invent a code" into a checkable
property, which is the property that matters for clinical documentation.

## What a real deployment would additionally require

This is a **prototype**, not a production clinical system. Before real use it
would need, at minimum:

- A signed Business Associate Agreement (BAA) with any model/API vendor.
- Encryption in transit and at rest; access controls and audit logging.
- A validated, NER-based de-identification pipeline (not regexes).
- Human-in-the-loop review of generated notes before they enter any record.
- Ongoing quality monitoring and drift/error-mode analysis in production.
- Clinical validation of accuracy and completeness against clinician-authored
  notes, not just synthetic gold labels.

## Honest scope statement

This demonstrates the full pipeline and the safety scaffolding a production
version would need. It does not prove production scale, real-PHI handling under a
BAA, or clinical validation. It should be described as such.
