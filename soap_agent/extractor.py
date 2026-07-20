"""
Extraction node: transcript + retrieved references -> raw SOAP dict.

This is where the LLM sits. The interface `extract(transcript, references,
prior_errors) -> dict` is backend-agnostic:

PRODUCTION backend (Anthropic API): the prompt includes the retrieved reference
chunks and, on a retry, the prior validation errors, instructing the model to
emit ONLY JSON matching the SOAP schema and to use ONLY ICD codes present in the
references. Sketch:

    import anthropic
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=SYSTEM_PROMPT,          # "emit only JSON; ground codes in refs"
        messages=[{"role": "user", "content": prompt}],
    )
    raw = json.loads(msg.content[0].text)

OFFLINE backend (default): a deterministic rule-based extractor that parses the
templated synthetic transcript. It is intentionally imperfect in a controlled
way (see `attempt`), so the agent's validate/retry loop actually fires and we
can prove the self-correction works rather than just asserting it does.
"""

from __future__ import annotations

import re


SYSTEM_PROMPT = (
    "You are a clinical scribe. Read the transcript and produce a SOAP note as "
    "strict JSON with keys: subjective, objective (list), assessment, plan "
    "(list), medications (list of {name,dose,frequency}), icd_candidates (list). "
    "Use ONLY ICD-10 codes that appear in the provided REFERENCES. Do not invent "
    "codes. If a required field is unknown, infer it from the transcript; never "
    "leave it blank. On a retry you will be shown prior errors; fix exactly those."
)


def _offline_extract(transcript: str, grounded_codes: set[str], attempt: int) -> dict:
    """Deterministic parser over the templated transcript.

    `attempt` simulates a model that gets it slightly wrong the first time:
    on attempt 0 it omits the `plan` (a required field) to trigger a real
    validation failure; on attempt >=1 it produces the complete note. This
    exercises the retry loop for real.
    """
    lines = [l.strip() for l in transcript.splitlines() if l.strip()]
    joined = " ".join(lines).lower()

    # Subjective: patient chief-complaint lines.
    complaints = [l.split(":", 1)[1].strip() for l in lines
                  if l.lower().startswith("patient:") and "brings you in" not in l.lower()]
    # drop identity chatter
    complaints = [c for c in complaints if not re.search(r"that's me|mrn|reach me|sounds good", c.lower())]
    subjective = "Patient reports " + "; ".join(complaints) if complaints else "Patient reports symptoms as described."

    # Objective: provider vitals / exam line.
    objective = []
    for l in lines:
        low = l.lower()
        if low.startswith("provider:") and re.search(r"temp|bp |glucose|hba1c|tender|straight leg|abdomen|heart rate|erythematous|edema|non-tender|strength", low):
            if "let me take a look" in low or "check your vitals" in low:
                continue
            body = l.split(":", 1)[1].strip()
            for part in re.split(r",| and ", body):
                p = part.strip().rstrip(".")
                if p and "let me take a look" not in p.lower() and "check your vitals" not in p.lower():
                    objective.append(p[0].upper() + p[1:])
    if not objective:
        objective = ["Exam findings documented"]

    # Assessment: provider "it looks like ..." line.
    assessment = "Clinical impression documented."
    for l in lines:
        m = re.search(r"it looks like (.+)", l, re.IGNORECASE)
        if m:
            assessment = m.group(1).strip().rstrip(".").capitalize()

    # Plan: provider plan line(s).
    plan = []
    for l in lines:
        low = l.lower()
        if l.lower().startswith("provider:") and re.search(r"rest and|continue|start you on|recheck|referral|avoid|reinforce|increase|physical therapy|ibuprofen|elevate|return if|follow up", low):
            body = l.split(":", 1)[1].strip()
            for part in re.split(r", and |, ", body):
                p = part.strip().rstrip(".")
                if p:
                    plan.append(p[0].upper() + p[1:])

    # Medications: "start you on X <dose>, <freq>".
    medications = []
    for l in lines:
        m = re.search(r"start you on ([A-Za-z]+) ([\d]+ ?mg),?\s*(.+)", l, re.IGNORECASE)
        if m:
            medications.append({
                "name": m.group(1).strip().capitalize(),
                "dose": m.group(2).strip(),
                "frequency": m.group(3).strip().rstrip("."),
            })

    # ICD: pick grounded codes whose description best matches the assessment.
    icd = _pick_grounded_code(assessment, joined, grounded_codes)

    note = {
        "subjective": subjective,
        "objective": objective,
        "assessment": assessment,
        "plan": plan,
        "medications": medications,
        "icd_candidates": icd,
    }

    # Controlled first-attempt imperfection to exercise the retry loop.
    if attempt == 0:
        note.pop("plan")   # omit a required field -> validation fails -> retry

    return note


def _pick_grounded_code(assessment: str, joined: str, grounded: set[str]) -> list[str]:
    """Choose an ICD code ONLY from the grounded (retrieved) set."""
    text = (assessment + " " + joined).lower()
    keyword_map = {
        "J06.9": ["upper respiratory", "viral", "sore throat"],
        "J02.9": ["pharyngitis"],
        "J00": ["common cold", "nasopharyngitis"],
        "I10": ["hypertension", "blood pressure"],
        "E11.9": ["diabetes"],
        "M54.50": ["back pain"],
        "K21.9": ["reflux", "gerd", "heartburn"],
        "J01.90": ["sinusitis"],
        "G43.909": ["migraine"],
        "N39.0": ["urinary"],
    }
    best = None
    best_hits = 0
    for code in grounded:
        hits = sum(1 for kw in keyword_map.get(code, []) if kw in text)
        if hits > best_hits:
            best, best_hits = code, hits
    return [best] if best else []


def extract(transcript: str, references: list[str], grounded_codes: set[str],
            prior_errors: list[str], attempt: int, *, offline: bool = True) -> dict:
    """Produce a raw SOAP dict. `prior_errors` present on retries."""
    if offline:
        return _offline_extract(transcript, grounded_codes, attempt)

    # --- production path ---
    import json
    import anthropic

    ref_block = "\n".join(f"- {r}" for r in references)
    err_block = ""
    if prior_errors:
        err_block = "\nYour previous attempt failed validation:\n" + \
                    "\n".join(f"- {e}" for e in prior_errors) + \
                    "\nFix exactly these problems."
    prompt = (
        f"REFERENCES (use only these ICD codes):\n{ref_block}\n\n"
        f"TRANSCRIPT:\n{transcript}\n\n"
        f"Produce the SOAP note as strict JSON.{err_block}"
    )
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    text = msg.content[0].text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    return json.loads(text)
