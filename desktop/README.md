# Folder RAG — Desktop App (PyWebView)

A native desktop window for Folder RAG. No browser tab, no `localhost:8501` URL — just an app.

## Architecture

```
desktop/
├── main.py                 # PyWebView entrypoint, opens the window, starts the API
├── api.py                  # FastAPI server (in-process) — bridge between UI and rag/
├── settings.py             # Persistent settings (API key, last folder, model, etc.)
├── ui/
│   ├── index.html          # The single-page UI (uses the design system)
│   ├── app.js              # UI logic, voice typing, calls /api/*
│   └── styles.css          # Imports ../../colors_and_type.css
└── README.md               # this file
```

The Python side runs a FastAPI server bound to `127.0.0.1` on a random port (so nobody else on your network can reach it). PyWebView opens a chromeless OS window pointing at it. From the user's POV, it's a normal app.

The UI uses the **Folder RAG design system** (see `../colors_and_type.css`) so it looks like a real product, not Streamlit.

## What's in this build

- **Folder picker** — native OS dialog via `window.pywebview.api.pick_folder()`
- **Index management** — rebuild button with live progress
- **Ask box with voice typing** — hold the mic button or press <kbd>Space</kbd> to dictate. Uses the browser's Web Speech API (Chromium engine inside PyWebView), so it works offline on macOS via the system recognizer; on Windows it uses the OS recognizer.
- **Settings panel** — edit OpenAI API key, Ollama URL, default model, retrieval mode. Settings persist to `~/.folder_rag/settings.json`.
- **Strict / fallback / pure-llm** modes
- **Sources panel** with page/slide/sheet/section + score
- **Dark theme by default** (override in settings)

## Run it

```bash
pip install -r requirements.txt
# (also needs the rag/ folder + its deps from rag-improvements/)
python -m desktop.main
```

First launch creates `~/.folder_rag/settings.json` with empty defaults. Open Settings (⌘,) and paste your OpenAI key, or stick with Ollama.

## Voice typing — how it works

The Web Speech API is enabled inside PyWebView's WebKit/Edge WebView. Pressing the mic button starts a `SpeechRecognition` session; interim results stream into the ask input as you speak; final result is committed when you stop. No audio leaves the machine on macOS — the OS dictation engine handles it. On Windows it uses Edge's online recognizer unless you've enabled offline dictation in Windows settings.

If voice doesn't work in your environment, the button is hidden gracefully (we feature-detect `webkitSpeechRecognition`).

## Settings persistence

`settings.py` writes a single JSON file:

```json
{
  "openai_api_key": "sk-...",
  "ollama_base_url": "http://localhost:11434",
  "ollama_model": "qwen3:8b",
  "default_backend": "Ollama",
  "default_retrieval_mode": "fallback",
  "last_folder": "/Users/me/research",
  "theme": "dark"
}
```

The file is `chmod 600` on POSIX systems so the API key isn't world-readable. On Windows it inherits user-only ACLs by default.

The Settings panel posts to `/api/settings` which validates and rewrites the file. The API key is **never logged** and is masked in the UI after save (`sk-···7a9F`).

## Packaging (later)

For a real distributable, wrap with PyInstaller:

```bash
pyinstaller --windowed --name "Folder RAG" --add-data "desktop/ui:ui" desktop/main.py
```

That gives you a `.app` on macOS or `.exe` on Windows. Code-signing not included.
