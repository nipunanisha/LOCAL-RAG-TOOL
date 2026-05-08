import os
from dataclasses import dataclass

# Bump this when chunking logic changes meaningfully — old indexes get rebuilt.
# 2.1 → adds parent-window grouping metadata (parent_id, parent_start, parent_end).
CHUNKER_VERSION = "2.1"


@dataclass
class AppConfig:
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    reranker_model: str = os.getenv(
        "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    ollama_timeout: int = int(os.getenv("OLLAMA_TIMEOUT", "300"))

    # retrieval
    initial_k: int = int(os.getenv("INITIAL_K", "30"))      # vector + bm25 each fetch this many
    rerank_k: int = int(os.getenv("RERANK_K", "20"))        # rerank this many
    final_k_default: int = int(os.getenv("FINAL_K", "5"))   # default top-k surfaced
    use_mmr: bool = os.getenv("USE_MMR", "1") == "1"
    mmr_lambda: float = float(os.getenv("MMR_LAMBDA", "0.7"))

    # query expansion
    use_query_expansion: bool = os.getenv("USE_QUERY_EXPANSION", "1") == "1"

    # fallback threshold on the cross-encoder score (not vector distance)
    fallback_threshold: float = float(os.getenv("FALLBACK_THRESHOLD", "1.0"))

    # chunking
    chunk_target_tokens: int = int(os.getenv("CHUNK_TARGET_TOKENS", "350"))
    chunk_min_tokens: int = int(os.getenv("CHUNK_MIN_TOKENS", "50"))
    chunk_overlap_sentences: int = int(os.getenv("CHUNK_OVERLAP_SENTENCES", "1"))

    # parent-child retrieval: embed small (child) chunks, surface larger (parent) windows.
    # parent text is the concatenation of `parent_window_chunks` consecutive children
    # within a single document, anchored on the matched child.
    use_parent_child: bool = os.getenv("USE_PARENT_CHILD", "1") == "1"
    parent_window_chunks: int = int(os.getenv("PARENT_WINDOW_CHUNKS", "3"))

    # ocr
    ocr_min_pixels: int = int(os.getenv("OCR_MIN_PIXELS", "200"))  # skip tiny images

    # ingest batching: chunks accumulated across files before a single embed/upsert call
    embed_batch_size: int = int(os.getenv("EMBED_BATCH_SIZE", "64"))


def get_chroma_dir_for_folder(folder_path: str) -> str:
    return os.path.join(folder_path, ".chroma")


def pick_device() -> str:
    """cuda > mps > cpu — never crashes."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"
