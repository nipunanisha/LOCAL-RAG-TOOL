"""
Build / update a per-folder Chroma index.

Idempotent: re-running on an unchanged folder is a no-op. Re-runs only
re-embed files whose mtime has changed, files new to the folder, or files
indexed by an older chunker version.
"""
from __future__ import annotations
import os
import hashlib
from typing import Tuple, Callable

import chromadb
from sentence_transformers import SentenceTransformer

from .config import AppConfig, CHUNKER_VERSION, pick_device

# Parent window size, in number of consecutive chunks. Embedding stays per-child;
# the `parent_id` groups children that should be expanded together at retrieval
# time. Centralised here so ingest and qa agree.
PARENT_WINDOW_CHUNKS_DEFAULT = 3
from .chunking import chunk_blocks
from .loaders import load_blocks, iter_files


def _doc_id(path: str) -> str:
    return hashlib.sha1(path.encode("utf-8")).hexdigest()[:16]


def _get_model(config: AppConfig) -> SentenceTransformer:
    return SentenceTransformer(config.embedding_model, device=pick_device())


def build_index_for_folder(
    folder: str,
    chroma_dir: str,
    config: AppConfig,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> Tuple[int, int]:
    """
    Returns (docs_indexed, chunks_added).
    on_progress(done, total, current_file) is called per file if provided.
    """
    os.makedirs(chroma_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=chroma_dir)
    collection = client.get_or_create_collection(name="docs")

    model = _get_model(config)

    # ---- read existing state ----
    existing = collection.get(include=["metadatas"])
    existing_ids = existing.get("ids", []) or []
    existing_meta = existing.get("metadatas", []) or []

    by_path: dict[str, dict] = {}  # path -> {"mtime": float, "version": str, "ids": [..]}
    for cid, meta in zip(existing_ids, existing_meta):
        if not meta:
            continue
        p = meta.get("source")
        if not p:
            continue
        info = by_path.setdefault(p, {"mtime": meta.get("mtime", 0), "version": meta.get("chunker_version", "1.0"), "ids": []})
        info["ids"].append(cid)

    current_files = list(iter_files(folder))
    current_paths = set(current_files)

    # delete vanished files
    for p in list(by_path.keys()):
        if p not in current_paths:
            collection.delete(ids=by_path[p]["ids"])
            del by_path[p]

    # ---- decide what to (re)index ----
    to_index: list[str] = []
    for p in current_files:
        info = by_path.get(p)
        mtime = os.path.getmtime(p)
        if info is None:
            to_index.append(p)
        elif mtime > info["mtime"] or info["version"] != CHUNKER_VERSION:
            to_index.append(p)

    total = len(to_index)
    docs_done = 0
    chunks_added = 0

    batch_size = max(1, int(getattr(config, "embed_batch_size", 64) or 64))
    window = getattr(config, "parent_window_chunks", PARENT_WINDOW_CHUNKS_DEFAULT) or PARENT_WINDOW_CHUNKS_DEFAULT
    window = max(1, int(window))

    # cross-file pending buffer; flushed when it reaches batch_size or at the end
    pend_ids: list[str] = []
    pend_texts: list[str] = []
    pend_metas: list[dict] = []

    def flush() -> int:
        if not pend_texts:
            return 0
        vectors = model.encode(
            pend_texts, normalize_embeddings=True, show_progress_bar=False, batch_size=batch_size
        ).tolist()
        collection.add(ids=pend_ids, documents=pend_texts, embeddings=vectors, metadatas=pend_metas)
        n = len(pend_texts)
        pend_ids.clear()
        pend_texts.clear()
        pend_metas.clear()
        return n

    for i, path in enumerate(to_index):
        rel = os.path.relpath(path, folder)
        if on_progress:
            on_progress(i, total, rel)
        try:
            blocks = load_blocks(path, ocr_min_pixels=config.ocr_min_pixels)
            if not blocks:
                continue
            chunks = chunk_blocks(
                blocks,
                target_tokens=config.chunk_target_tokens,
                min_tokens=config.chunk_min_tokens,
                overlap_sentences=config.chunk_overlap_sentences,
            )
            if not chunks:
                continue

            mtime = os.path.getmtime(path)
            ext = os.path.splitext(path)[1].lower().lstrip(".")
            did = _doc_id(path)

            # remove old chunks for this file before queuing new ones
            if path in by_path:
                collection.delete(ids=by_path[path]["ids"])

            for idx, c in enumerate(chunks):
                p_index = idx // window
                p_start = p_index * window
                p_end = min(p_start + window - 1, len(chunks) - 1)
                m = {
                    "source": path,
                    "rel_path": rel,
                    "ext": ext,
                    "chunk": idx,
                    "mtime": mtime,
                    "char_len": len(c.text),
                    "doc_id": did,
                    "chunker_version": CHUNKER_VERSION,
                    "parent_id": f"{did}::parent::{p_index}",
                    "parent_start": p_start,
                    "parent_end": p_end,
                    "parent_window": window,
                }
                for k in ("page", "slide", "section", "sheet", "kind", "row_start", "row_end"):
                    if k in c.meta and c.meta[k] not in (None, ""):
                        m[k] = c.meta[k]
                pend_ids.append(f"{did}::chunk::{idx}")
                pend_texts.append(c.text)
                pend_metas.append(m)

                if len(pend_texts) >= batch_size:
                    chunks_added += flush()

            docs_done += 1
        except Exception as e:
            print(f"[ingest] {path}: {e}")

    chunks_added += flush()

    if on_progress:
        on_progress(total, total, "")

    return docs_done, chunks_added
