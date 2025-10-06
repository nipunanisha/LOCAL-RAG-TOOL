## RAG App (Folder-based)

This is a minimal Retrieval Augmented Generation (RAG) app with a Streamlit UI. You can:

- Select a folder containing documents (.pdf, .txt, .md, .docx, .ppt, .csv, .html)
- (Re)index them into a local Chroma vector store with sentence-transformers
- Ask questions against the selected folder
- Choose the LLM backend: OpenAI or Ollama

### Quickstart

1. Create and activate a Python 3.10+ venv.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. (Optional) Set environment variables. Copy `.env.example` to `.env` and fill values.
4. Run the app:

```bash
streamlit run app.py
```

### Configuration

- Embeddings model: sentence-transformers `all-MiniLM-L6-v2`
- Vector DB: Chroma (stored per-folder under a hidden subfolder `.chroma`)
- Supported docs: PDF, TXT, MD, DOCX, PPT, CSV, HTML
- LLM backends:
  - OpenAI: requires `OPENAI_API_KEY`
  - Ollama: requires local Ollama running at `http://localhost:11434`

### Notes
- Each folder you select will have its own persistent index at `<folder>/.chroma`.
- Click "Rebuild Index" to re-index the folder after adding/removing documents.
- Answers include the top sources used from retrieval.

### Features

- Select a folder containing various document types (.pdf, .txt, .md, .docx, .ppt, .csv, .html).
- Rebuild index for the selected folder with a local Chroma vector store.
- Ask questions against documents in the selected folder.
- Choose between OpenAI or Ollama LLM backends.
- Configure retrieval modes: Strict RAG, RAG with fallback, or Pure LLM.
- Persistent per-folder indexing stored in a hidden `.chroma` subfolder.
- Displays CUDA availability and GPU name if available.

### Usage

- Enter the folder path containing your documents in the sidebar.
- Click "Rebuild Index" to index or reindex documents.
- Enter your question and click "Ask" to get answers with source references.
- Configure LLM backend and retrieval mode via sidebar options.

### Requirements

- Python 3.10+
- Dependencies listed in `requirements.txt`
- Environment variables setup for OpenAI API key or running Ollama locally.

### Running the app

```bash
streamlit run app.py
```

