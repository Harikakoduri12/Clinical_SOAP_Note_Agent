"""
The agent: a cyclic graph that retrieves, extracts, validates, and self-corrects.

    retrieve --> extract --> validate --> (valid?) --> END
                    ^                          |
                    └──────── no (retry) ──────┘   [bounded by MAX_ATTEMPTS]

Why a graph with a cycle and not a linear chain: the validate->extract backward
edge is a cycle, and cycles are exactly what LangGraph adds over a plain chain
(a DAG, which cannot loop). The loop is the self-correction: a failed Pydantic
validation writes its errors into state, and the next extract attempt sees those
errors and fixes precisely what broke. MAX_ATTEMPTS bounds the loop so it can
never spin forever.

This module ships the graph in pure Python so it runs with no dependency. The
node functions and the routing are written so that wiring them into the real
`langgraph.graph.StateGraph` is mechanical — see `build_langgraph()` at the
bottom for the exact real-LangGraph construction (guarded so its absence doesn't
break the offline run).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict

from .retriever import Retriever
from .extractor import extract
from .schema import validate_draft, SOAPNote
from .guardrails import redact_transcript, detect_phi_leak


MAX_ATTEMPTS = 3


class AgentState(TypedDict, total=False):
    transcript: str            # de-identified transcript (post input-guardrail)
    known_names: list[str]     # for name redaction
    planted_phi: list[str]     # for the output leak guardrail
    retrieved_chunks: list[str]
    grounded_codes: set[str]
    references: list[str]
    draft: dict
    validation_errors: list[str]
    attempts: int
    note: dict | None
    status: str                # "ok" | "failed_validation" | "phi_leak"
    trace: list[str]           # human-readable log of the run


# ---- Nodes -----------------------------------------------------------------

def node_retrieve(state: AgentState, retriever: Retriever, k: int = 3) -> AgentState:
    results = retriever.retrieve(state["transcript"], k=k)
    state["retrieved_chunks"] = [r.chunk.text for r in results]
    state["references"] = [r.chunk.text for r in results]
    state["grounded_codes"] = retriever.grounded_codes(results)
    state.setdefault("trace", []).append(
        f"retrieve: {len(results)} chunks, grounded codes={sorted(state['grounded_codes'])}"
    )
    return state


def node_extract(state: AgentState) -> AgentState:
    state["attempts"] = state.get("attempts", 0) + 1
    draft = extract(
        transcript=state["transcript"],
        references=state["references"],
        grounded_codes=state["grounded_codes"],
        prior_errors=state.get("validation_errors", []),
        attempt=state["attempts"] - 1,
        offline=False,
    )
    state["draft"] = draft
    state["trace"].append(
        f"extract: attempt {state['attempts']} "
        f"(prior_errors={len(state.get('validation_errors', []))})"
    )
    return state


def node_validate(state: AgentState) -> AgentState:
    outcome = validate_draft(state["draft"])
    if not outcome.ok:
        state["validation_errors"] = outcome.errors
        state["trace"].append(f"validate: FAIL -> {outcome.errors}")
        return state

    # structural validity passed; now enforce ICD grounding (agent-level check)
    ungrounded = [c for c in outcome.note.icd_candidates
                  if c not in state["grounded_codes"]]
    if ungrounded:
        state["validation_errors"] = [
            f"icd_candidates: {ungrounded} not grounded in retrieved references"
        ]
        state["trace"].append(f"validate: FAIL (ungrounded codes {ungrounded})")
        return state

    state["validation_errors"] = []
    state["note"] = outcome.note.model_dump()
    state["trace"].append("validate: PASS")
    return state


# ---- Conditional edge ------------------------------------------------------

def route_after_validate(state: AgentState) -> str:
    """The conditional edge. Returns the name of the next node."""
    if not state.get("validation_errors"):
        return "guardrail_out"
    if state["attempts"] >= MAX_ATTEMPTS:
        return "give_up"
    return "extract"      # <-- the cycle: back to extract with errors in state


def node_guardrail_out(state: AgentState) -> AgentState:
    """Output guardrail: no planted PHI may survive into the final note."""
    import json
    note_text = json.dumps(state["note"])
    leak = detect_phi_leak(note_text, state.get("planted_phi", []))
    if not leak.clean:
        state["status"] = "phi_leak"
        state["note"] = None
        state["trace"].append(f"guardrail_out: PHI LEAK {leak.leaks} -> blocked")
    else:
        state["status"] = "ok"
        state["trace"].append("guardrail_out: clean")
    return state


# ---- Driver (pure-Python graph executor) -----------------------------------

@dataclass
class AgentResult:
    note: dict | None
    status: str
    attempts: int
    trace: list[str] = field(default_factory=list)


def run_agent(transcript: str, *, known_names: list[str] | None = None,
              planted_phi: list[str] | None = None,
              retriever: Retriever | None = None, k: int = 3) -> AgentResult:
    """Execute the cyclic graph over one transcript.

    Applies the INPUT guardrail (redaction) first, then walks the graph.
    """
    retriever = retriever or Retriever()

    # INPUT guardrail: de-identify before anything else touches the text.
    redaction = redact_transcript(transcript, known_names=known_names or [])

    state: AgentState = {
        "transcript": redaction.text,
        "known_names": known_names or [],
        "planted_phi": planted_phi or [],
        "attempts": 0,
        "validation_errors": [],
        "note": None,
        "status": "",
        "trace": [f"input_guardrail: redacted {redaction.removed}"],
    }

    # retrieve (once), then the extract/validate cycle
    state = node_retrieve(state, retriever, k=k)
    node = "extract"
    while True:
        if node == "extract":
            state = node_extract(state)
            node = "validate"
        elif node == "validate":
            state = node_validate(state)
            node = route_after_validate(state)
        elif node == "guardrail_out":
            state = node_guardrail_out(state)
            break
        elif node == "give_up":
            state["status"] = "failed_validation"
            state["trace"].append(
                f"give_up: still invalid after {state['attempts']} attempts"
            )
            break

    return AgentResult(
        note=state.get("note"),
        status=state.get("status", ""),
        attempts=state.get("attempts", 0),
        trace=state.get("trace", []),
    )


# ---- Real LangGraph construction (used if langgraph is installed) ----------

def build_langgraph(retriever: Retriever, k: int = 3):
    """Return a compiled LangGraph StateGraph with identical topology.

    Kept import-guarded so the offline demo never requires langgraph. This is
    the exact wiring you'd ship; the node functions above are reused verbatim.
    """
    from langgraph.graph import StateGraph, END

    g = StateGraph(AgentState)
    g.add_node("retrieve", lambda s: node_retrieve(s, retriever, k))
    g.add_node("extract", node_extract)
    g.add_node("validate", node_validate)
    g.add_node("guardrail_out", node_guardrail_out)

    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "extract")
    g.add_edge("extract", "validate")
    g.add_conditional_edges(
        "validate",
        route_after_validate,
        {"extract": "extract", "guardrail_out": "guardrail_out", "give_up": END},
    )
    g.add_edge("guardrail_out", END)
    return g.compile()