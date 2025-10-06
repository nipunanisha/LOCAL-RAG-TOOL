import os
import time
import json
import streamlit as st
import torch

from rag.config import AppConfig, get_chroma_dir_for_folder

from rag.qa import answer_question
from rag.ingest import build_index_for_folder

st.set_page_config(page_title="Folder RAG", page_icon="🧠", layout="wide")
st.title("🧠 Folder RAG - Ask your documents")

config = AppConfig()

CONFIG_PATH = os.path.join(os.path.expanduser("~"), ".folder_rag_config.json")

def load_last_folder_path() -> str:
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("last_folder", "")
    except Exception:
        return ""

def save_last_folder_path(path: str) -> None:
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump({"last_folder": path}, f)
    except Exception:
        pass

with st.sidebar:
    st.header("Settings")
    if "folder" not in st.session_state:
        st.session_state["folder"] = load_last_folder_path()

    def on_folder_change():
        save_last_folder_path(st.session_state["folder"])

    st.text_input(
        "Folder path containing docs",
        value=st.session_state["folder"],
        placeholder="E:/hobby/docs or /path/to/folder",
        key="folder",
        on_change=on_folder_change,
    )

    llm_backend = st.selectbox("LLM backend", ["OpenAI", "Ollama"], index=1)
    ollama_model = None
    if llm_backend == "Ollama":
        ollama_model = st.text_input(
            "Ollama model", value="qwen3:8b",
            help="Example: qwen3:8b, qwen2.5:7b, llama3.1, mistral, etc."
        )

    top_k = st.slider("Top K passages", 1, 12, 5)
    rebuild = st.button("Rebuild Index")

    retrieval_option = st.selectbox(
        "Retrieval mode",
        ["Strict RAG", "RAG → fallback to general LLM", "Pure LLM"],
        index=1
    )
    retrieval_mode = (
        "strict" if retrieval_option == "Strict RAG" 
        else "fallback" if retrieval_option == "RAG → fallback to general LLM" 
        else "pure_llm"
    )

st.write("Supported: .pdf, .txt, .md, .docx • Index stored in the folder as `.chroma`.")

folder = st.session_state.get("folder", "")

if folder:
    if not os.path.isdir(folder):
        st.error("Folder does not exist. Please enter a valid path.")
        st.stop()

    chroma_dir = get_chroma_dir_for_folder(folder)

    if rebuild:
        with st.status("Indexing documents...", expanded=True) as status:
            start = time.time()
            num_docs, num_chunks = build_index_for_folder(folder, chroma_dir, config)
            status.update(label=f"Indexed {num_docs} docs into {num_chunks} chunks", state="complete")
            st.success(f"Index built in {time.time()-start:.1f}s")

    st.subheader("Ask a question")
    question = st.text_input("Your question")
    ask = st.button("Ask")

    if ask and question.strip():
        with st.spinner("Thinking..."):
            result = answer_question(
                question=question.strip(),
                folder_path=folder,
                chroma_dir=chroma_dir,
                top_k=top_k,
                backend=llm_backend,
                config=config,
                ollama_model=ollama_model,
                retrieval_mode=retrieval_mode
            )
        st.markdown("**Answer**")
        st.write(result["answer"])
        st.markdown("**Sources**")
        for src in result["sources"]:
            st.caption(f"{src['source']} (score={src['score']:.3f})")
else:
    st.info("Enter a documents folder path in the sidebar to begin.")

st.write("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    st.write("GPU name:", torch.cuda.get_device_name(0))
else:
    st.write("Running on CPU")
