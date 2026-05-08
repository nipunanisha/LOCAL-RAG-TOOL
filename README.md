# RAG Tool — AI File Explorer

A native Windows desktop app that turns any folder on your computer into a searchable, question-answerable knowledge base. Point it at a folder full of PDFs, Word docs, slides, spreadsheets, or notes and ask questions in plain English — answers come back with citations to the exact page, slide, sheet, or section.

Runs fully local with [Ollama](https://ollama.ai), or in the cloud with the OpenAI API. Your choice per query.

---

## Features

- **Folder-as-a-knowledge-base** — index any local folder; sub-folders included
- **Hybrid retrieval** — vector search (semantic) + BM25 (keyword) fused with reciprocal rank fusion
- **Cross-encoder reranking** for precision on the final shortlist
- **Parent-child chunking** — embeddings are over small chunks for accuracy, the LLM sees the larger surrounding window for context
- **MMR diversity** so multiple sources show up instead of three near-duplicates
- **Three answer modes** — `strict` (only from your files), `fallback` (files first, general knowledge if needed), `pure_llm` (no retrieval)
- **Citations with page/slide/sheet/section pins** — click through to verify
- **Voice typing** — hold the mic button or press <kbd>Space</kbd> to dictate questions
- **Native OS folder picker**, dark UI, no browser tab
- **Settings persisted locally** — API keys stay on your machine

## Supported file types

PDF, DOCX, PPTX, XLSX, TXT, MD, HTML, and more (anything the loaders in `rag/loaders.py` can parse).

---

## System requirements

| Requirement | Version |
|---|---|
| OS | Windows 10/11, 64-bit |
| Python | **3.12 specifically** (the shipped `.pyd` modules are `cp312-win_amd64`) |
| RAM | 8 GB minimum, 16 GB recommended |
| Disk | ~5 GB for dependencies (PyTorch + models) |
| GPU (optional) | NVIDIA with CUDA 12.1 — auto-detected, falls back to CPU |

For local LLM inference: install [Ollama](https://ollama.ai) and pull a model (e.g. `ollama pull llama3.1`).
For cloud LLM: an OpenAI API key.

---

## Install

### Option 1 — Automated installer (recommended)

```cmd
setup.bat
```

This will:
1. Check for / install Python 3.12
2. Create a virtual environment in `RagTool/` next to the project
3. Install all dependencies from `requirements.txt`
4. Save your install path to `install_config.txt`

Takes 5–15 minutes depending on bandwidth (PyTorch + CUDA wheels are large).

### Option 2 — Manual

```powershell
python -m venv RagTool
.\RagTool\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Run

After installation:

```cmd
RAGTOOL.exe
```

Or from the activated venv:

```powershell
python -m desktop.main
```

The app opens a native window titled **AI File Explorer** at 1180×780.

### First-time setup inside the app

1. Open **Settings** and either:
   - Paste your **OpenAI API key**, or
   - Set the **Ollama base URL** (default `http://localhost:11434`) and pick a model
2. Click **Pick Folder** and choose what you want to index
3. Click **Build Index** — first build takes a while (it downloads embedding + reranker models on first run, ~500 MB)
4. Ask away

Settings are stored in `~/.folder_rag/settings.json` (chmod 600 / user-only ACL).

---

## Project layout

```
RAGTool/
├── desktop/                    # PyWebView desktop shell
│   ├── main.py                 # entrypoint — boots FastAPI + opens window
│   ├── *.pyd                   # compiled api.py, settings.py, __init__.py
│   ├── ui/
│   │   ├── index.html          # single-page UI
│   │   ├── app.js              # UI logic + voice typing
│   │   └── styles.css          # imports ../../colors_and_type.css
│   └── README.md               # desktop-specific notes
│
├── rag/                        # retrieval + LLM core (compiled to .pyd)
│   ├── config.pyd              # AppConfig, device picker
│   ├── loaders.pyd             # parsers for PDF/DOCX/PPTX/XLSX/...
│   ├── chunking.pyd            # parent-child chunking
│   ├── ingest.pyd              # build the Chroma index
│   └── qa.pyd                  # hybrid retrieve → rerank → MMR → LLM
│
├── assets/                     # logos and icons
├── colors_and_type.css         # design system (typography + palette)
├── requirements.txt
├── setup.bat / setup.ps1       # Windows installer
├── RAGTOOL.exe                 # launcher executable
├── RAGTOOL.ico                 # app icon
└── install_config.txt          # written by the installer
```

The Python core ships as compiled `.pyd` extension modules (Cython). Source `.py` files are not in the published build — only `desktop/main.py` remains as plain Python because it's the entrypoint.

---

## How it works

```
question
   │
   ├─► query expansion (2 paraphrases via the LLM)
   │
   ├─► hybrid retrieve  ──► vector search (sentence-transformers)
   │                    └─► BM25 keyword search
   │
   ├─► reciprocal rank fusion
   │
   ├─► cross-encoder rerank (top ~50 → top ~10)
   │
   ├─► MMR diversity selection
   │
   ├─► parent-window expansion (small chunks → larger context)
   │
   └─► LLM with cited context  ──► answer + sources [1] [2] [3]
```

Citations are anchored on the matched **child** chunk so page/slide pins stay precise, even though the LLM reads the larger parent window.

### Retrieval modes

| Mode | Behavior |
|---|---|
| `strict` | Answer only from your files. If the answer isn't there, replies *"I don't know based on these files."* |
| `fallback` | Prefer your files, but use general knowledge if context is thin (marked with `(general knowledge)`) |
| `pure_llm` | Skip retrieval entirely — straight LLM call |

---

## Architecture

The app is a single process:

1. `desktop/main.py` finds a free localhost port
2. Spawns a thread running **FastAPI + Uvicorn** on `127.0.0.1:<random>`
3. Opens a chromeless **PyWebView** window pointing at that URL
4. JS in the UI calls `/api/*` endpoints; native dialogs go through `window.pywebview.api.*` (folder picker, etc.)

Network is bound to localhost only — nothing on your LAN can reach the API.

---

## Troubleshooting

**`Expected str, got tuple` when asking a question**
Old build issue with Cython type annotations. Fixed in current builds. If it returns, recompile from source after fixing return-type annotations.

**`ImportError: DLL load failed` on startup**
You're running on the wrong Python version. The `.pyd` files are built for **Python 3.12** specifically. Check `python --version`.

**Ollama timeout**
Increase `ollama_timeout` in settings, or pick a smaller/faster model (`llama3.1:8b` instead of `70b`).

**Indexing is slow on first run**
First build downloads `sentence-transformers/all-MiniLM-L6-v2` (embeddings) and a cross-encoder reranker. ~500 MB, one-time. Subsequent builds reuse them.

**No GPU detected**
The app auto-falls-back to CPU. To use GPU you need an NVIDIA card and the CUDA 12.1 PyTorch wheel (already pinned in `requirements.txt`).

---

## License

Proprietary. All rights reserved.
