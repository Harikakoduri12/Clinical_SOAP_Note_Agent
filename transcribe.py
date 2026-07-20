"""
Speech-to-text stage.

The public interface is `transcribe(audio_path) -> str`, matching how you'd call
Whisper. Two modes:

OFFLINE (default): there is no audio in this demo — synthetic encounters are born
as text — so offline mode reads the paired transcript file. This keeps the whole
pipeline runnable with no model weights.

PRODUCTION (documented): real Whisper, e.g.

    import whisper
    model = whisper.load_model("base")            # or "small"/"medium"
    result = model.transcribe(audio_path)         # returns text + segments
    return result["text"]

or the faster-whisper / API variants. The rest of the pipeline is identical
because it only ever consumes the returned string.
"""

from __future__ import annotations

import os


def transcribe(source: str, *, offline: bool = True) -> str:
    """Return a transcript string for the given audio (or text) source."""
    if offline:
        # In the demo, `source` is a path to a .txt transcript (the stand-in
        # for Whisper output). Read and return it verbatim.
        if os.path.exists(source):
            with open(source, "r", encoding="utf-8") as fh:
                return fh.read()
        # Or `source` may already be the transcript text itself.
        return source

    # --- production path (requires `pip install openai-whisper` + audio) ---
    import whisper  # noqa: F401  (imported lazily so offline needs no dep)

    model = whisper.load_model(os.environ.get("WHISPER_MODEL", "base"))
    result = model.transcribe(source)
    return result["text"]
