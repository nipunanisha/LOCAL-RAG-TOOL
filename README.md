# RAG Tool — AI File Explorer

A native Windows desktop app that turns any folder on your computer into a searchable, question-answerable knowledge base. Point it at a folder full of PDFs, Word docs, slides, spreadsheets, or notes — then ask questions in natural language and get cited answers.

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
- **GPU support** — auto-detects NVIDIA CUDA 12.1, falls back to CPU

## Supported file types

PDF, DOCX, PPTX, XLSX, TXT, MD, HTML, and more (anything the loaders in `rag/loaders.pyd` can parse).

---

## System requirements

| Requirement | Version |
|---|---|
| OS | Windows 10/11, 64-bit |
| Python | **3.12 specifically** (the shipped `.pyd` modules are `cp312-win_amd64`) |
| RAM | 8 GB minimum, 16 GB recommended |
| Disk | ~5 GB for dependencies (PyTorch + models) |
| GPU (optional) | NVIDIA with CUDA 12.1 — auto-detected, falls back to CPU |

### Dependencies

For local LLM inference: install [Ollama](https://ollama.ai) and pull a model (e.g. `ollama pull llama3.1`).  
For cloud LLM: an OpenAI API key.

---

## Quick Start

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

## Project Structure

```
LOCAL-RAG-TOOL/
├── desktop/                         # PyWebView desktop shell
│   ├── main.py                      # entrypoint — boots FastAPI + opens window
│   ├── api.cp312-win_amd64.pyd      # compiled FastAPI server (bridge to RAG core)
│   ├── settings.cp312-win_amd64.pyd # compiled settings manager
│   ├── __init__.cp312-win_amd64.pyd # compiled module init
│   ├── ui/                          # Single-page web UI
│   │   ├── index.html               # UI shell
│   │   ├── app.js                   # UI logic + voice typing
│   │   └── styles.css               # styling (imports ../../colors_and_type.css)
│   ├── requirements.txt             # desktop-specific deps
│   └── README.md                    # desktop-specific documentation
│
├── rag/                             # Retrieval + LLM core (compiled to .pyd)
│   ├── config.cp312-win_amd64.pyd         # AppConfig, device picker
│   ├── loaders.cp312-win_amd64.pyd        # parsers for PDF/DOCX/PPTX/XLSX/...
│   ├── chunking.cp312-win_amd64.pyd       # parent-child chunking
│   ├── ingest.cp312-win_amd64.pyd         # build the Chroma vector index
│   └── qa.cp312-win_amd64.pyd             # hybrid retrieve → rerank → MMR → LLM
│
├── assets/                          # logos and icons
├── colors_and_type.css              # design system (typography + palette)
├── requirements.txt                 # main dependencies (PyTorch, sentence-transformers, etc.)
├── setup.bat / setup.ps1            # Windows installer scripts
├── RAGTOOL.exe                      # launcher executable
├── RAGTOOL.ico                      # app icon
├── install_config.txt               # written by installer (install path)
└── README.md                        # this file
```

**Note:** The Python core ships as compiled `.pyd` extension modules (Cython/NumPy compiled code). Source `.py` files are not in the published build — only `desktop/main.py` remains as plain Python because it's the entrypoint.

---

## How It Works

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

### Retrieval Modes

| Mode | Behavior |
|---|---|
| `strict` | Answer only from your files. If the answer isn't there, replies *"I don't know based on these files."* |
| `fallback` | Prefer your files, but use general knowledge if context is thin (marked with `(general knowledge)`) |
| `pure_llm` | Skip retrieval entirely — straight LLM call |

---

## Architecture

The app is a single-process desktop application:

1. **`desktop/main.py`** finds a free localhost port and starts the server
2. **FastAPI + Uvicorn** thread runs on `127.0.0.1:<random port>` (localhost only, not accessible from LAN)
3. **PyWebView** opens a chromeless native OS window pointing at the local API
4. **JavaScript UI** calls `/api/*` endpoints for backend operations
5. **Native dialogs** (folder picker, file operations) go through `window.pywebview.api.*` bridge

Network is bound to localhost only — nothing on your LAN can reach the API.

### Component Breakdown

- **Desktop Shell (`desktop/`)**: PyWebView window + FastAPI server binding
- **RAG Core (`rag/`)**: Document loading, chunking, embeddings, vector search, reranking, LLM integration
- **UI (`desktop/ui/`)**: Single-page app (HTML/JS/CSS) with dark theme, voice input, settings panel
- **Design System (`colors_and_type.css`)**: Shared typography and color palette

---

## Dependencies

Key libraries (see `requirements.txt` for full list):

- **[ChromaDB](https://www.trychroma.com/)** — vector database for embeddings
- **[Sentence Transformers](https://www.sbert.net/)** — semantic embeddings (all-MiniLM-L6-v2)
- **[PyTorch](https://pytorch.org/)** — deep learning (with optional CUDA 12.1 support)
- **[OpenAI API](https://openai.com/api/)** — cloud LLM option
- **[Ollama](https://ollama.ai)** — local LLM runtime
- **[FastAPI](https://fastapi.tiangolo.com/)** + **[Uvicorn](https://www.uvicorn.org/)** — web server
- **[PyWebView](https://pywebview.flowrl.com/)** — native desktop window
- **[rank-bm25](https://github.com/dorianbrown/rank_bm25)** — keyword search
- **[python-docx](https://python-docx.readthedocs.io/)** — Word document parsing
- **[PyPDF](https://github.com/py-pdf/pypdf)** — PDF parsing
- **[python-pptx](https://python-pptx.readthedocs.io/)** — PowerPoint parsing
- **[openpyxl](https://openpyxl.readthedocs.io/)** — Excel parsing
- **[PyMuPDF](https://pymupdf.readthedocs.io/)** — PDF/document rendering

---

## Troubleshooting

**`Expected str, got tuple` when asking a question**
Old build issue with Cython type annotations. Fixed in current builds. If it persists, recompile from source.

**`ImportError: DLL load failed` on startup**
You're running on the wrong Python version. The `.pyd` files are built for **Python 3.12** specifically.  
Check: `python --version`

**Ollama timeout / slow responses**
- Increase `ollama_timeout` in settings
- Pick a smaller/faster model: `llama3.1:8b` instead of `70b`
- Ensure Ollama is running: `ollama serve`

**Indexing is slow on first run**
First build downloads `sentence-transformers/all-MiniLM-L6-v2` (embeddings) and a cross-encoder reranker (~500 MB, one-time). Subsequent builds reuse cached models.

**No GPU detected**
The app auto-falls-back to CPU. To use GPU, you need:
- NVIDIA graphics card
- CUDA 12.1 PyTorch wheel (already pinned in `requirements.txt`)
- NVIDIA drivers installed

**Folder indexing / vector search is very slow**
- Check RAM usage (target: 16 GB+)
- Try a smaller subset of files first
- Ensure no antivirus scanning is blocking `~/.folder_rag/` directory
- On SSD: indexing is significantly faster

**API key not saving / settings lost**
- Check write permissions to `~/.folder_rag/`
- On Windows, verify user-only ACL on `settings.json` (if manually created)
- Try: Settings → Save again

---

## Development & Customization

### Running from source

```powershell
# Activate venv
.\RagTool\Scripts\Activate.ps1

# Run directly (skips RAGTOOL.exe launcher)
python -m desktop.main
```

### Modifying the UI

Edit files in `desktop/ui/` and refresh the browser window (`F5` or `Cmd+R`) to see changes. No rebuild needed for HTML/CSS/JS.

### Recompiling Python modules

The `.pyd` files are pre-compiled Cython extensions. To recompile:
1. Install Cython and a C compiler (MSVC on Windows)
2. Modify the `.py` source files
3. Run the build script (if available) or use `cython` + `distutils`

---

## License

Proprietary. All rights reserved.

---

## Support & Feedback

For issues, suggestions, or contributions, please refer to the GitHub repository issues page.
