#!/usr/bin/env python3
"""
Evidence-surfaced clarification demo — the complications-inference case.

Shows the interaction the WellSky engineer described: for a missing field, the
system surfaces the relevant transcript line AND the question, then defers the
clinical decision to the clinician.

    python scripts/evidence_demo.py
    python scripts/evidence_demo.py --live
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from soap_agent.coverage import CoverageReport, Gap                  # noqa: E402
from soap_agent.evidence_clarification import (                      # noqa: E402
    build_clarifications, record_answer,
)

BAR = "=" * 78

# The clinician mentions lethargy tied to diabetes, but the
# recorded primary diagnosis is kidney disease. The model can recite this line
# but cannot interpret whether it answers "complications from primary diagnosis".
TRANSCRIPT = """
Provider: Let's review your chronic conditions today.
Patient:  My kidney disease has been the main concern lately.
Provider: Right, that's the primary issue we're managing.
Patient:  I've also been really tired and sluggish all the time.
Provider: That lethargy — I believe that's related to your diabetes.
Patient:  Okay. And what about my blood pressure medication?
Provider: We'll keep that the same for now.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    args = ap.parse_args()
    offline = not args.live

    # the coverage layer (deterministic) has already flagged this required field
    # as missing. we hand it in directly here to focus on the surfacing step.
    gap = Gap(
        section="assessment",
        field="complications_primary_dx",
        description="complications resulting from the primary diagnosis",
    )
    report = CoverageReport(total_required=1, captured=[], gaps=[gap])

    print(f"\n{BAR}")
    print("  EVIDENCE-SURFACED CLARIFICATION  (complications-inference case)")
    print(BAR)
    print("  primary diagnosis on record:  Kidney disease")
    print("  missing required field:       complications from the primary diagnosis")
    print()
    print("  The model recorded kidney disease as primary, so it cannot connect")
    print("  'lethargy from diabetes' to this field. It can SURFACE the line. Only")
    print("  the clinician can DECIDE whether it belongs here.")

    items = build_clarifications(report, TRANSCRIPT, offline=offline)

    for item in items:
        print(f"\n{BAR}")
        print("  INTERACTION OBJECT")
        print(BAR)
        print(f"    field_id      : {item.field_id}")
        print(f"    field_name    : {item.field_name}")
        print(f"    status        : {item.status}")
        print(f"    evidence surfaced from the transcript:")
        for e in item.relevant_transcript_evidence:
            print(f"        \"{e}\"")
        print(f"    question      : {item.clarification_question}")
        print()

        # be honest about what was actually surfaced — do not hardcode a claim.
        found_lethargy = any("lethargy" in e.lower()
                             for e in item.relevant_transcript_evidence)
        if found_lethargy:
            print("    <- the search surfaced the lethargy line. It did NOT decide")
            print("       the line answers the field. That is the clinician's call.")
        else:
            print("    <- NOTE: the search did NOT surface the lethargy line here.")
            print("       Even semantic search can miss the relevant line, which is")
            print("       exactly why the clinician still reviews rather than trusts")
            print("       the surfaced evidence. Surfacing is an aid, not an oracle.")

        # simulate the clinician confirming
        record_answer(item,
                      "Lethargy, attributed by provider to comorbid diabetes.",
                      confirmed=True)
        print()
        print(f"    clinician_answer      : {item.clinician_answer}")
        print(f"    confirmed_by_clinician: {item.confirmed_by_clinician}")
        print(f"    status                : {item.status}")

    print(f"\n{BAR}")
    print("  THE INTERACTION OBJECT AS JSON (what a UI would consume)")
    print(BAR)
    print(json.dumps([i.to_dict() for i in items], indent=2))

    print(f"\n{BAR}")
    print("  WHO DID WHAT")
    print(BAR)
    print("""
    schema      detected the gap                        (deterministic)
    system      surfaced the relevant transcript line   (search)
    model       phrased the question                    (language)
    CLINICIAN   decided lethargy belongs in this field  (clinical judgment)

    The system found the needle. The clinician decided it was a needle. That line
    is the entire safety argument — surfacing is a search problem, interpreting is
    a clinical one, and the model never crosses from the first into the second.

    Note also: because the clinician's answer is not stated verbatim in the
    transcript, the downstream audit will mark it grounded_to_transcript=False,
    provenance=clinician. Correct: it's traceable to the clinician, not invented.
""")


if __name__ == "__main__":
    main()
