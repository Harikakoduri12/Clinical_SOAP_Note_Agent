#!/usr/bin/env python3
"""Walk one synthetic encounter through the full pipeline, printing each stage.

Usage: python scripts/demo.py [seed]
"""



import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.synthetic import generate_encounter
from soap_agent.transcribe import transcribe
from soap_agent.agent import run_agent


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    e = generate_encounter(seed=seed)

    print("=" * 70)
    print(f"ENCOUNTER {e.encounter_id}  (patient: {e.patient_name})")
    print("=" * 70)

    print("\n[1] WHISPER (offline: transcript stand-in)")
    transcript = transcribe(e.transcript, offline=True)
    print("    first line:", transcript.splitlines()[0])
    print("    (contains raw PHI: name, MRN, DOB, phone, email)")

    print("\n[2-6] AGENT: redact -> retrieve -> extract -> validate/retry -> guardrail")
    result = run_agent(
        transcript,
        known_names=[e.patient_name, e.provider_name],
        planted_phi=e.planted_phi,
    )
    for line in result.trace:
        print("    •", line)

    print(f"\n[7] RESULT  status={result.status}  attempts={result.attempts}")
    print(json.dumps(result.note, indent=2))

    print("\n[GOLD] for comparison")
    print(json.dumps(e.gold_note, indent=2))


if __name__ == "__main__":
    main()
