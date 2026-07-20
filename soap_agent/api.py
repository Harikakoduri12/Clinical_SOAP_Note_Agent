"""
FastAPI service wrapping the pipeline.

Endpoints:
  GET  /health      -> liveness
  POST /transcribe  -> audio/text source -> transcript (Whisper stage)
  POST /soap        -> transcript (+ optional known names/phi) -> SOAP note

Run:  uvicorn soap_agent.api:app --reload
Docs: http://localhost:8000/docs  (auto-generated from the Pydantic models)

Request/response bodies are typed Pydantic models, so validation and OpenAPI
docs come for free — the same schema discipline used inside the agent.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

try:
    from fastapi import FastAPI, HTTPException
except ImportError:  # keep the module importable without fastapi installed
    FastAPI = None

from .agent import run_agent
from .transcribe import transcribe
from .schema import SOAPNote


class TranscribeRequest(BaseModel):
    source: str = Field(..., description="Path to audio/transcript, or raw text.")


class TranscribeResponse(BaseModel):
    transcript: str


class SOAPRequest(BaseModel):
    transcript: str
    known_names: list[str] = Field(default_factory=list)
    planted_phi: list[str] = Field(default_factory=list,
        description="Known identifiers to enforce zero-leak against (testing).")
    top_k: int = 3


class SOAPResponse(BaseModel):
    status: str
    attempts: int
    note: SOAPNote | None
    trace: list[str]


def create_app():
    if FastAPI is None:
        raise RuntimeError("fastapi not installed")
    app = FastAPI(title="Clinical SOAP Agent", version="1.0.0")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/transcribe", response_model=TranscribeResponse)
    def do_transcribe(req: TranscribeRequest):
        text = transcribe(req.source, offline=True)
        return TranscribeResponse(transcript=text)

    @app.post("/soap", response_model=SOAPResponse)
    def do_soap(req: SOAPRequest):
        result = run_agent(
            req.transcript,
            known_names=req.known_names,
            planted_phi=req.planted_phi,
            k=req.top_k,
        )
        note = SOAPNote(**result.note) if result.note else None
        if result.status == "phi_leak":
            raise HTTPException(status_code=422, detail="Blocked: PHI leak detected")
        return SOAPResponse(
            status=result.status,
            attempts=result.attempts,
            note=note,
            trace=result.trace,
        )

    return app


app = create_app() if FastAPI is not None else None
