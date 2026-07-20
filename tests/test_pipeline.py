"""End-to-end and unit tests. Run: python -m pytest tests/ -v"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.synthetic import generate_encounter, generate_dataset
from soap_agent.agent import run_agent
from soap_agent.retriever import Retriever
from soap_agent.schema import validate_draft
from soap_agent.guardrails import redact_transcript, detect_phi_leak


def test_schema_accepts_valid_note():
    out = validate_draft({
        "subjective": "Sore throat three days.",
        "objective": ["Temp 100.1F"],
        "assessment": "Viral URI",
        "plan": ["Rest"],
    })
    assert out.ok


def test_schema_reports_missing_plan():
    out = validate_draft({
        "subjective": "Sore throat three days.",
        "objective": ["Temp 100.1F"],
        "assessment": "Viral URI",
    })
    assert not out.ok
    assert any("plan" in e for e in out.errors)


def test_schema_rejects_malformed_icd():
    out = validate_draft({
        "subjective": "Sore throat three days.",
        "objective": ["Temp 100.1F"],
        "assessment": "Viral URI",
        "plan": ["Rest"],
        "icd_candidates": ["NOTACODE"],
    })
    assert not out.ok


def test_retry_loop_fires_and_recovers():
    e = generate_encounter(seed=0)
    res = run_agent(e.transcript,
                    known_names=[e.patient_name, e.provider_name],
                    planted_phi=e.planted_phi)
    # attempt 1 omits plan (fails), attempt 2 recovers
    assert res.status == "ok"
    assert 1 <= res.attempts <= 3
    assert any("PASS" in t for t in res.trace)

def test_icd_is_grounded():
    e = generate_encounter(seed=0)
    r = Retriever()
    res = run_agent(e.transcript, known_names=[e.patient_name, e.provider_name],
                    planted_phi=e.planted_phi, retriever=r)
    grounded = r.grounded_codes(r.retrieve(res_transcript(e, r), k=3))
    for code in res.note["icd_candidates"]:
        assert code in grounded


def res_transcript(e, r):
    return redact_transcript(e.transcript,
                             known_names=[e.patient_name, e.provider_name]).text


def test_input_guardrail_redacts_phi():
    e = generate_encounter(seed=2)
    rep = redact_transcript(e.transcript,
                            known_names=[e.patient_name, e.provider_name])
    # planted phone/mrn/email should be gone from redacted text
    assert "[PHONE]" in rep.text
    assert "[MRN]" in rep.text or "[NAME]" in rep.text


def test_output_has_no_phi_leak():
    import json
    for e in generate_dataset(5):
        res = run_agent(e.transcript,
                        known_names=[e.patient_name, e.provider_name],
                        planted_phi=e.planted_phi)
        leak = detect_phi_leak(json.dumps(res.note), e.planted_phi)
        assert leak.clean, f"leak in {e.encounter_id}: {leak.leaks}"


def test_leak_detector_catches_planted_value():
    leak = detect_phi_leak("Note about Jane Roe", ["Jane Roe"])
    assert not leak.clean and "Jane Roe" in leak.leaks
