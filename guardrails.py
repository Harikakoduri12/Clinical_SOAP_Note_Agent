"""
PHI guardrails — two of them, deliberately at two points in the pipeline.

INPUT guardrail  (pre-LLM): redact identifiers from the transcript before the
                 model ever sees them. Defense-in-depth: even with synthetic
                 data we treat it as if it were real.

OUTPUT guardrail (post-generation): scan the final note for any identifier that
                 leaked through. Leakage must be zero; a leak is a hard failure.

Production note: a real system would use a clinical NER de-identifier such as
Microsoft Presidio or a spaCy clinical model rather than regexes. The regex
layer here is deterministic and testable, and — critically — because our
synthetic data injects *known* PHI (via Faker), we can measure redaction recall
exactly against the values we planted. See THREAT_MODEL.md for the honest
statement of what this does and does not guarantee.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# Deterministic identifier patterns. Order matters: more specific first.
_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("MRN", re.compile(r"\bMRN[:#]?\s*\d{4,10}\b", re.IGNORECASE)),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("PHONE", re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")),
    ("DOB", re.compile(r"\b(?:0?[1-9]|1[0-2])/(?:0?[1-9]|[12]\d|3[01])/(?:19|20)\d\d\b")),
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
]


@dataclass
class RedactionReport:
    text: str
    removed: dict[str, int] = field(default_factory=dict)
    # what was actually replaced, so tests can confirm exact recall
    removed_values: list[str] = field(default_factory=list)


def redact_transcript(text: str, known_names: list[str] | None = None) -> RedactionReport:
    """Replace identifiers with typed placeholders like [PHONE].

    `known_names` lets callers pass the exact injected patient/provider names so
    name redaction is exact rather than guessed (names are the hardest category
    for a pure regex, which is why production uses NER).
    """
    removed: dict[str, int] = {}
    removed_values: list[str] = []
    out = text

    for label, pattern in _PATTERNS:
        def _sub(m):
            removed_values.append(m.group(0))
            removed[label] = removed.get(label, 0) + 1
            return f"[{label}]"
        out = pattern.sub(_sub, out)

    if known_names:
        for name in known_names:
            for token in filter(None, name.split()):
                nm = re.compile(rf"\b{re.escape(token)}\b")
                def _subn(m):
                    removed_values.append(m.group(0))
                    removed["NAME"] = removed.get("NAME", 0) + 1
                    return "[NAME]"
                out = nm.sub(_subn, out)

    return RedactionReport(text=out, removed=removed, removed_values=removed_values)


@dataclass
class LeakReport:
    clean: bool
    leaks: list[str] = field(default_factory=list)


def detect_phi_leak(output_text: str, planted_phi: list[str]) -> LeakReport:
    """Output guardrail: does any planted PHI value survive into the note?

    `planted_phi` is the list of exact identifier strings we injected into the
    source encounter (names, phone, MRN, ...). If any appears verbatim in the
    output, that's a leak and a hard failure.
    """
    leaks = []
    low = output_text.lower()
    for value in planted_phi:
        v = value.strip()
        if len(v) < 3:
            continue
        if v.lower() in low:
            leaks.append(v)
    return LeakReport(clean=(len(leaks) == 0), leaks=leaks)
