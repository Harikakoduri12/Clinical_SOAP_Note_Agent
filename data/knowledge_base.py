"""
The medical reference library, sliced into retrievable chunks.

In production this would be the full ICD-10 catalog + a drug reference loaded
from disk. Here it's a compact, realistic subset covering the presentations our
synthetic encounters use, so the whole RAG path is exercisable end to end.

Each chunk carries the ICD-10 code(s) it describes. Grounding is enforced by
checking that every code in a generated note appears in the chunks retrieved
for that encounter.
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Chunk:
    id: str
    text: str
    codes: list[str] = field(default_factory=list)


KNOWLEDGE_BASE: list[Chunk] = [
    Chunk(
        id="kb_uri",
        text=(
            "J06.9 Acute upper respiratory infection, unspecified. A viral "
            "illness presenting with sore throat, nasal congestion, rhinorrhea, "
            "cough, and low-grade fever. Self-limiting; supportive care with "
            "rest, fluids, and analgesics such as acetaminophen."
        ),
        codes=["J06.9"],
    ),
    Chunk(
        id="kb_cold",
        text=(
            "J00 Acute nasopharyngitis (common cold). Rhinorrhea, sneezing, "
            "sore throat, mild malaise without significant fever. Supportive "
            "management only."
        ),
        codes=["J00"],
    ),
    Chunk(
        id="kb_pharyngitis",
        text=(
            "J02.9 Acute pharyngitis, unspecified. Throat pain and erythema. "
            "Consider streptococcal testing if fever, tonsillar exudate, and "
            "absence of cough are present."
        ),
        codes=["J02.9"],
    ),
    Chunk(
        id="kb_htn",
        text=(
            "I10 Essential (primary) hypertension. Persistently elevated blood "
            "pressure. Management includes lifestyle modification and "
            "antihypertensives; monitor blood pressure and medication adherence."
        ),
        codes=["I10"],
    ),
    Chunk(
        id="kb_t2dm",
        text=(
            "E11.9 Type 2 diabetes mellitus without complications. Chronic "
            "hyperglycemia managed with metformin, dietary changes, and "
            "glucose monitoring. Track HbA1c and screen for complications."
        ),
        codes=["E11.9"],
    ),
    Chunk(
        id="kb_sinusitis",
        text=(
            "J01.90 Acute sinusitis, unspecified. Facial pain/pressure, nasal "
            "congestion, and purulent discharge. Most cases are viral; "
            "antibiotics reserved for persistent or severe bacterial cases."
        ),
        codes=["J01.90"],
    ),
    Chunk(
        id="kb_lowbackpain",
        text=(
            "M54.50 Low back pain, unspecified. Mechanical back pain without "
            "radiculopathy. Managed with activity modification, NSAIDs, and "
            "physical therapy; imaging only if red-flag features present."
        ),
        codes=["M54.50"],
    ),
    Chunk(
        id="kb_gerd",
        text=(
            "K21.9 Gastro-esophageal reflux disease without esophagitis. "
            "Heartburn and regurgitation. Managed with lifestyle changes and "
            "proton pump inhibitors such as omeprazole."
        ),
        codes=["K21.9"],
    ),
    Chunk(
        id="kb_migraine",
        text=(
            "G43.909 Migraine, unspecified, not intractable, without status "
            "migrainosus. Recurrent throbbing headache with photophobia and "
            "nausea. Managed with triptans and trigger avoidance."
        ),
        codes=["G43.909"],
    ),
    Chunk(
        id="kb_uti",
        text=(
            "N39.0 Urinary tract infection, site not specified. Dysuria, "
            "urinary frequency, and urgency. Confirmed by urinalysis; treated "
            "with appropriate antibiotics such as nitrofurantoin."
        ),
        codes=["N39.0"],
    ),
]


def all_codes() -> set[str]:
    codes: set[str] = set()
    for c in KNOWLEDGE_BASE:
        codes.update(c.codes)
    return codes
