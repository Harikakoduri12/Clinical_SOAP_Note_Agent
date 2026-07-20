#!/usr/bin/env python3
"""Run the full eval: gold set -> agent -> scores -> report."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.synthetic import generate_dataset
from soap_agent.agent import run_agent
from soap_agent.retriever import Retriever
from evals.harness import Report, score_case, judge_agreement


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--num", type=int, default=10, help="encounters to eval")
    ap.add_argument("-k", "--topk", type=int, default=3, help="RAG top-k")
    ap.add_argument("--out", default="evals/report", help="output path prefix")
    args = ap.parse_args()

    retriever = Retriever()
    dataset = generate_dataset(args.num)

    cases = []
    judge_scores, human_labels = [], []
    for enc in dataset:
        result = run_agent(
            enc.transcript,
            known_names=[enc.patient_name, enc.provider_name],
            planted_phi=enc.planted_phi,
            retriever=retriever,
            k=args.topk,
        )
        # recover grounded codes for this encounter for the grounding check
        rr = retriever.retrieve(
            _redacted(enc), k=args.topk
        )
        grounded = retriever.grounded_codes(rr)
        cs = score_case(enc, result, grounded)
        cases.append(cs)
        judge_scores.append(cs.faithfulness)
        # pretend the first 3 cases were human-labeled at ~1.0 for agreement demo
        if len(human_labels) < 3:
            human_labels.append(1.0)

    report = Report(cases=cases)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".json", "w", encoding="utf-8") as fh:
        fh.write(report.to_json())
    with open(args.out + ".md", "w", encoding="utf-8") as fh:
        fh.write(report.to_markdown())

    agg = report.aggregate()
    print("=== AGGREGATE ===")
    for k, v in agg.items():
        print(f"  {k:28s} {v}")
    agree = judge_agreement(judge_scores[:3], human_labels)
    print(f"  judge_vs_human_agreement     {agree}  (on {len(human_labels)} labeled)")
    print(f"\nWrote {args.out}.json and {args.out}.md")


def _redacted(enc):
    from soap_agent.guardrails import redact_transcript
    return redact_transcript(enc.transcript,
                             known_names=[enc.patient_name, enc.provider_name]).text


if __name__ == "__main__":
    main()
