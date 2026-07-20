"""
RAG retriever over the medical knowledge base.

WHY THIS FILE WAS REWRITTEN
---------------------------
The first version scored chunks by bag-of-words cosine similarity — literal word
overlap. That is not retrieval. It is a lookup table, and it fails on the exact
problem retrieval exists to solve: **patients do not speak in ICD-10**.

Measured failures of the keyword version, against this same knowledge base:

    "I think I had a heart attack"  -> returned the UPPER RESPIRATORY chunk,
                                       matching on the word "a". Confidently wrong.
    "I cannot catch my breath"      -> returned NOTHING. Zero chunks, zero
                                       grounded codes, model left with no evidence.

A retriever that hands a respiratory code to a cardiac complaint is worse than no
retriever, because the grounding check downstream will happily approve it — the
code IS in the retrieved set. The failure is silent.

The fix is semantic search: embed text into vectors that encode *meaning* rather
than spelling, so "heart attack" lands near "myocardial infarction" despite
sharing no characters.

HOW IT WORKS
------------
1. At startup, embed every knowledge-base chunk once (sentence-transformers,
   all-MiniLM-L6-v2 — small, fast, CPU-only).
2. At query time, embed the transcript and take cosine similarity against the
   pre-computed chunk vectors.
3. Return top-k above a floor. The floor matters: without it an unrelated query
   still returns *something*, which is how the keyword version handed back a
   respiratory chunk for a cardiac complaint.

Vectors are L2-normalized, so cosine similarity is a plain dot product. At this
corpus size (tens of chunks) a direct numpy dot beats a FAISS index — FAISS earns
its keep at 10^5+ vectors, not 10^1. The interface is unchanged, so swapping in
FAISS later touches only `_score_all`.

FALLBACK
--------
If sentence-transformers is unavailable this degrades to the old lexical scorer
and says so loudly on stderr. It never silently pretends to be semantic.
"""

from __future__ import annotations

import math
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.knowledge_base import KNOWLEDGE_BASE, Chunk  # noqa: E402


#: Cosine floor. Below this, a chunk is not evidence — it is noise.
MIN_SCORE = 0.20

_MODEL_NAME = "all-MiniLM-L6-v2"

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _cosine_counter(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    shared = set(a) & set(b)
    dot = sum(a[t] * b[t] for t in shared)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float


class Retriever:
    """Semantic top-k retriever over the clinical knowledge base.

    Public interface unchanged from the keyword version: `retrieve(query, k)` and
    `grounded_codes(results)`. Nothing downstream in the agent moved.
    """

    def __init__(self, corpus: list[Chunk] | None = None,
                 min_score: float = MIN_SCORE, semantic: bool = True):
        self.corpus = corpus if corpus is not None else KNOWLEDGE_BASE
        self.min_score = min_score
        self._model = None
        self._emb = None
        self.semantic = False

        if semantic:
            self._try_load_semantic()

        if not self.semantic:
            self._doc_vecs = {c.id: Counter(_tokenize(c.text)) for c in self.corpus}

    def _try_load_semantic(self) -> None:
        """Load the embedding model. Fall back loudly, never silently.

        Catches Exception, not just ImportError: the library can be installed and
        still fail (no network on first run, model not cached, disk full). A
        retriever that dies on startup is worse than one that degrades and says so.
        """
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(_MODEL_NAME)
            self._emb = self._model.encode(
                [c.text for c in self.corpus],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            self.semantic = True
        except Exception as exc:                       # noqa: BLE001
            print(f"[retriever] SEMANTIC MODE UNAVAILABLE ({type(exc).__name__}) "
                  f"-> falling back to LEXICAL matching.\n"
                  f"[retriever] Lexical cannot bridge 'heart attack' -> "
                  f"'myocardial infarction'. Retrieval quality is degraded.\n"
                  f"[retriever] Fix: pip install sentence-transformers "
                  f"(first run downloads ~90MB).",
                  file=sys.stderr)

    def _score_all(self, query: str) -> list[float]:
        """Similarity of the query against every chunk.

        Swap this one method for a FAISS index search; nothing else changes.
        """
        if self.semantic:
            q = self._model.encode([query], normalize_embeddings=True,
                                   show_progress_bar=False)
            return (self._emb @ q[0]).tolist()      # normalized -> dot == cosine
        qv = Counter(_tokenize(query))
        return [_cosine_counter(qv, self._doc_vecs[c.id]) for c in self.corpus]

    def retrieve(self, query: str, k: int = 3) -> list[RetrievalResult]:
        scores = self._score_all(query)
        results = [RetrievalResult(chunk=c, score=float(s))
                   for c, s in zip(self.corpus, scores)]
        results.sort(key=lambda r: r.score, reverse=True)
        # the floor is the point: no evidence beats wrong evidence
        return [r for r in results[:k] if r.score >= self.min_score]

    def grounded_codes(self, results: list[RetrievalResult]) -> set[str]:
        """ICD codes present in the retrieved chunks.

        A generated note may only emit codes from this set. If retrieval returns
        nothing this is empty — the correct outcome for a complaint the knowledge
        base does not cover.
        """
        codes: set[str] = set()
        for r in results:
            codes.update(r.chunk.codes)
        return codes
