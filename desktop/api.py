"""
FastAPI server — bridge between the UI and the rag/ package.

Endpoints:
  GET  /                      → ui/index.html
  GET  /static/*              → ui assets
  GET  /api/settings          → current settings (api key masked)
  POST /api/settings          → update settings (api key in clear, server stores it)
  POST /api/index             → kick off indexing for a folder (sync; small folders OK)
  POST /api/ask                → ask a question
  GET  /api/health            → cuda/ollama/openai availability
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Any
from fastapi import FastAPI, Body, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# import the rag package — assumes you've placed rag-improvements/rag/ at repo root as rag/
try:
    from rag.config import AppConfig, get_chroma_dir_for_folder, pick_device
    from rag.ingest import build_index_for_folder
    from rag.qa import answer_question
    RAG_OK = True
except Exception as e:  # noqa
    RAG_OK = False
    _rag_err = str(e)

from .settings import load_settings, save_settings, mask_key, OPENAI_MODELS

UI_DIR = Path(__file__).parent / "ui"

app = FastAPI(title="AI File Explorer", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=str(UI_DIR)), name="static")

# stash for the JS bridge
_pywebview_window = None
def set_pywebview_window(win):
    global _pywebview_window
    _pywebview_window = win


@app.get("/")
def root():
    return FileResponse(str(UI_DIR / "index.html"))


# ---------------------------- settings ----------------------------
class SettingsIn(BaseModel):
    openai_api_key: str | None = None
    openai_model: str | None = None
    ollama_base_url: str | None = None
    ollama_model: str | None = None
    default_backend: str | None = None
    default_retrieval_mode: str | None = None
    default_top_k: int | None = None
    last_folder: str | None = None
    theme: str | None = None


@app.get("/api/settings")
def get_settings():
    s = load_settings()
    out = dict(s)
    out["openai_api_key"] = mask_key(s.get("openai_api_key", ""))
    out["openai_api_key_set"] = bool(s.get("openai_api_key"))
    out["openai_models"] = OPENAI_MODELS
    return out


@app.post("/api/settings")
def post_settings(body: SettingsIn):
    incoming = {k: v for k, v in body.model_dump().items() if v is not None}
    # never accept the masked placeholder back
    if "openai_api_key" in incoming and "···" in incoming["openai_api_key"]:
        incoming.pop("openai_api_key")
    saved = save_settings(incoming)
    out = dict(saved)
    out["openai_api_key"] = mask_key(saved.get("openai_api_key", ""))
    out["openai_api_key_set"] = bool(saved.get("openai_api_key"))
    return out


# ---------------------------- health ----------------------------
@app.get("/api/health")
def health():
    info: dict[str, Any] = {"rag_ok": RAG_OK, "device": "cpu"}
    if not RAG_OK:
        info["rag_error"] = _rag_err
        return info

    try:
        info["device"] = pick_device()
    except Exception:
        pass

    s = load_settings()
    # check ollama
    try:
        import requests
        r = requests.get(f"{s['ollama_base_url']}/api/tags", timeout=1.5)
        info["ollama"] = r.ok
    except Exception:
        info["ollama"] = False
    info["openai_key_set"] = bool(s.get("openai_api_key"))
    return info


# ---------------------------- index ----------------------------
class IndexIn(BaseModel):
    folder: str


@app.post("/api/index")
def post_index(body: IndexIn):
    if not RAG_OK:
        raise HTTPException(500, f"rag package not available: {_rag_err}")
    folder = os.path.expanduser(body.folder)
    if not os.path.isdir(folder):
        raise HTTPException(400, "folder does not exist")
    s = load_settings()
    cfg = _config_from_settings(s)
    chroma = get_chroma_dir_for_folder(folder)
    docs, chunks = build_index_for_folder(folder, chroma, cfg)
    save_settings({"last_folder": folder})
    return {"docs": docs, "chunks": chunks, "folder": folder}


# ---------------------------- ask ----------------------------
class AskIn(BaseModel):
    folder: str
    question: str
    top_k: int | None = None
    backend: str | None = None
    retrieval_mode: str | None = None


@app.post("/api/ask")
def post_ask(body: AskIn):
    if not RAG_OK:
        raise HTTPException(500, f"rag package not available: {_rag_err}")
    folder = os.path.expanduser(body.folder)
    if not os.path.isdir(folder):
        raise HTTPException(400, "folder does not exist")
    if not body.question.strip():
        raise HTTPException(400, "empty question")

    s = load_settings()
    cfg = _config_from_settings(s)
    chroma = get_chroma_dir_for_folder(folder)

    backend = body.backend or s.get("default_backend") or "Ollama"
    mode = body.retrieval_mode or s.get("default_retrieval_mode") or "fallback"
    top_k = body.top_k or s.get("default_top_k") or 5

    if backend == "OpenAI" and not s.get("openai_api_key"):
        raise HTTPException(400, "openai api key not set — open settings to add it")

    try:
        result = answer_question(
            question=body.question.strip(),
            folder_path=folder,
            chroma_dir=chroma,
            top_k=top_k,
            backend=backend,
            config=cfg,
            ollama_model=s.get("ollama_model"),
            openai_model=s.get("openai_model"),
            retrieval_mode=mode,
        )
    except Exception as e:
        raise HTTPException(500, str(e))
    return result


def _config_from_settings(s: dict) -> "AppConfig":
    cfg = AppConfig()
    if s.get("openai_api_key"):
        cfg.openai_api_key = s["openai_api_key"]
    if s.get("openai_model"):
        cfg.openai_model = s["openai_model"]
    if s.get("ollama_base_url"):
        cfg.ollama_base_url = s["ollama_base_url"]
    return cfg
