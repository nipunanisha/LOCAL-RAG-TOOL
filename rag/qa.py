"""
Hybrid retrieval (vector + BM25) → cross-encoder rerank → MMR diversity →
parent-window expansion → LLM.

Embeddings are over small ("child") chunks for retrieval precision; the LLM
sees the larger ("parent") context window each child belongs to. The LLM is
asked to cite sources as [1], [2] matching the order of the returned
`sources` list — citations are still anchored on the matched child so
page/slide/section pins stay precise.
"""
from __future__ import annotations
import re
from typing import Literal
from functools import lru_cache

import requests
import chromadb

from .config import AppConfig, pick_device


# ----------------------------- model singletons -----------------------------
@lru_cache(maxsize=2)
def _embedder(name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name, device=pick_device())


@lru_cache(maxsize=2)
def _reranker(name: str):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(name, device=pick_device())


# ----------------------------- retrieval -----------------------------
def _vector_search(query: str, collection, embedder, top: int) -> list[dict]:
    qvec = embedder.encode([query], normalize_embeddings=True).tolist()
    res = collection.query(
        query_embeddings=qvec,
        n_results=top,
        include=["documents", "metadatas", "distances"],
    )
    out = []
    if not res["ids"] or not res["ids"][0]:
        return out
    for i in range(len(res["ids"][0])):
        out.append({
            "id": res["ids"][0][i],
            "text": res["documents"][0][i],
            "meta": res["metadatas"][0][i] or {},
            "vec_distance": float(res["distances"][0][i]),
        })
    return out


def _bm25_search(query: str, collection, top: int) -> list[dict]:
    """Pull whole corpus from chroma (small enough for personal folders),
    score with BM25, return top."""
    from rank_bm25 import BM25Okapi

    all_docs = collection.get(include=["documents", "metadatas"])
    ids = all_docs.get("ids", []) or []
    docs = all_docs.get("documents", []) or []
    metas = all_docs.get("metadatas", []) or []
    if not docs:
        return []

    tokenized = [_tokenize(d) for d in docs]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(_tokenize(query))
    ranked = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)[:top]
    return [
        {"id": ids[i], "text": docs[i], "meta": metas[i] or {}, "bm25_score": float(scores[i])}
        for i in ranked if scores[i] > 0
    ]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _rrf_fuse(*ranked_lists: list[dict], k: int = 60) -> list[dict]:
    """Reciprocal rank fusion — score-scale agnostic."""
    by_id: dict[str, dict] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked):
            cur = by_id.setdefault(item["id"], {"id": item["id"], "text": item["text"], "meta": item["meta"], "rrf": 0.0})
            cur["rrf"] += 1.0 / (k + rank + 1)
    return sorted(by_id.values(), key=lambda x: x["rrf"], reverse=True)


def _rerank(query: str, candidates: list[dict], reranker, top: int) -> list[dict]:
    if not candidates:
        return []
    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs).tolist()
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    return candidates[:top]


# ----------------------------- parent-child expansion -----------------------------
def _expand_to_parents(matched: list[dict], collection) -> list[dict]:
    """For each matched child chunk, fetch its sibling chunks (same parent_id)
    and concatenate them in document order. Preserves the rank order of the
    matched parents (one parent appears once, in the position of its earliest
    matched child).

    Each output dict carries:
      - text: concatenated parent text
      - meta: metadata of the matched child (so citations point at the most
              relevant slice — page/slide/section is taken from the hit, not
              the surrounding window)
      - rerank_score / rrf: copied from the matched child for downstream sort
      - id: the parent_id (so it appears unique to dedup logic)
      - children: [chunk indices included] — useful for debug
    """
    if not matched:
        return []
    seen: dict[str, dict] = {}
    order: list[str] = []
    for m in matched:
        pid = (m.get("meta") or {}).get("parent_id")
        if not pid:
            # no parent metadata (legacy index) — fall through as-is
            seen[m["id"]] = m
            order.append(m["id"])
            continue
        if pid in seen:
            continue  # dedupe — earliest-ranked child wins
        seen[pid] = m
        order.append(pid)

    # Batch-fetch all parent windows in one Chroma call. Group by doc_id so
    # `where` is satisfiable; chroma's `where` only supports one $eq per field.
    by_doc: dict[str, list[dict]] = {}
    for key in order:
        m = seen[key]
        meta = m.get("meta") or {}
        if not meta.get("parent_id"):
            continue
        by_doc.setdefault(meta["doc_id"], []).append(m)

    # For each doc, fetch all chunks once, then slice locally.
    doc_cache: dict[str, dict] = {}
    for did in by_doc:
        try:
            res = collection.get(
                where={"doc_id": did},
                include=["documents", "metadatas"],
            )
            ids = res.get("ids", []) or []
            docs = res.get("documents", []) or []
            metas = res.get("metadatas", []) or []
            # index by chunk number
            by_chunk: dict[int, dict] = {}
            for i, meta in enumerate(metas):
                if not meta:
                    continue
                by_chunk[int(meta.get("chunk", i))] = {
                    "id": ids[i],
                    "text": docs[i],
                    "meta": meta,
                }
            doc_cache[did] = by_chunk
        except Exception:
            doc_cache[did] = {}

    out: list[dict] = []
    for key in order:
        m = seen[key]
        meta = m.get("meta") or {}
        pid = meta.get("parent_id")
        if not pid:
            out.append(m)
            continue
        did = meta["doc_id"]
        p_start = int(meta.get("parent_start", meta.get("chunk", 0)))
        p_end = int(meta.get("parent_end", meta.get("chunk", 0)))
        by_chunk = doc_cache.get(did, {})
        pieces = []
        included = []
        for ci in range(p_start, p_end + 1):
            sib = by_chunk.get(ci)
            if sib and sib["text"]:
                pieces.append(sib["text"])
                included.append(ci)
        if not pieces:
            # fall back to the matched child alone
            pieces = [m["text"]]
            included = [int(meta.get("chunk", 0))]
        out.append({
            "id": pid,
            "text": "\n\n".join(pieces),
            "meta": meta,            # citation anchored on the matched child
            "rerank_score": m.get("rerank_score", 0.0),
            "rrf": m.get("rrf", 0.0),
            "children": included,
            "child_text": m["text"],  # kept for MMR (small embedding)
        })
    return out


def _mmr(query_vec, candidates: list[dict], embedder, k: int, lam: float) -> list[dict]:
    """Maximum Marginal Relevance over already-reranked candidates."""
    if len(candidates) <= k:
        return candidates
    import numpy as np
    cand_vecs = embedder.encode([c["text"] for c in candidates], normalize_embeddings=True)
    qv = np.array(query_vec)
    selected: list[int] = []
    remaining = set(range(len(candidates)))
    # seed with rank-1
    selected.append(0)
    remaining.discard(0)
    while len(selected) < k and remaining:
        best_i, best_score = None, -1e9
        for i in remaining:
            sim_q = float(np.dot(cand_vecs[i], qv))
            sim_sel = max(float(np.dot(cand_vecs[i], cand_vecs[j])) for j in selected)
            score = lam * sim_q - (1 - lam) * sim_sel
            if score > best_score:
                best_score, best_i = score, i
        selected.append(best_i)
        remaining.discard(best_i)
    return [candidates[i] for i in selected]


# ----------------------------- query expansion -----------------------------
def _expand_query(question: str, backend: str, config: AppConfig, ollama_model: str | None, openai_model: str | None = None) -> list[str]:
    """Ask the LLM for 2 alternate phrasings. Cheap insurance."""
    sys = (
        "Rewrite the user question into 2 different, concise phrasings that mean the same thing. "
        "Vary the wording and synonyms. Output only the 2 rewrites, one per line, no numbering, no quotes."
    )
    msgs = [{"role": "system", "content": sys}, {"role": "user", "content": question}]
    try:
        if backend == "OpenAI":
            text, _ = call_openai(msgs, config, model=openai_model)
        else:
            text, _ = call_ollama(msgs, config, model=ollama_model)
        rewrites = [l.strip(" -•\"'") for l in text.splitlines() if l.strip()]
        return [question] + rewrites[:2]
    except Exception:
        return [question]


# ----------------------------- LLM calls -----------------------------
def call_openai(messages: list[dict], config: AppConfig, model: str | None = None) -> str:
    from openai import OpenAI
    if not config.openai_api_key:
        raise ValueError("OPENAI_API_KEY not set.")
    client = OpenAI(api_key=config.openai_api_key)
    resp = client.chat.completions.create(
        model=model or config.openai_model,
        messages=messages,
        temperature=0.2,
    )
    text = resp.choices[0].message.content or ""
    # Try provider-reported usage, fall back to local tiktoken estimation
    usage = {}
    try:
        u = getattr(resp, "usage", None)
        if u is None:
            try:
                u = resp.get("usage", None)
            except Exception:
                u = None

        def _get_u(field: str) -> int:
            try:
                v = getattr(u, field)
                if v is not None:
                    return int(v)
            except Exception:
                pass
            try:
                v = u.get(field)
                if v is not None:
                    return int(v)
            except Exception:
                pass
            try:
                v = u[field]
                if v is not None:
                    return int(v)
            except Exception:
                pass
            return 0

        if u:
            pt = _get_u("prompt_tokens")
            ct = _get_u("completion_tokens")
            tt = _get_u("total_tokens")
            if tt == 0:
                tt = pt + ct
            usage = {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": tt}
    except Exception:
        usage = {}

    if not usage:
        # estimate via tiktoken if available
        try:
            import tiktoken
            mdl = model or config.openai_model
            try:
                enc = tiktoken.encoding_for_model(mdl)
            except Exception:
                enc = tiktoken.get_encoding("cl100k_base")
            prompt_text = "\n".join([m.get("content", "") for m in messages])
            p_toks = len(enc.encode(prompt_text))
            c_toks = len(enc.encode(text))
            usage = {"prompt_tokens": int(p_toks), "completion_tokens": int(c_toks), "total_tokens": int(p_toks + c_toks)}
        except Exception:
            # fallback heuristic: approximate tokens by word count if tiktoken not present
            import re
            p_words = len(re.findall(r"\w+", "\n".join([m.get("content", "") for m in messages])))
            c_words = len(re.findall(r"\w+", text))
            # approximate: 1 token ≈ 1 word (safe fallback)
            usage = {"prompt_tokens": int(p_words), "completion_tokens": int(c_words), "total_tokens": int(p_words + c_words)}

    return text, usage


def call_ollama(messages: list[dict], config: AppConfig, model: str | None) -> str:
    url = f"{config.ollama_base_url}/api/chat"
    payload = {"model": model or "llama3.1", "messages": messages, "stream": False, "options": {"temperature": 0.2}}
    try:
        r = requests.post(url, json=payload, timeout=config.ollama_timeout)
        r.raise_for_status()
        text = r.json().get("message", {}).get("content", "")
    except requests.exceptions.ReadTimeout:
        raise RuntimeError(f"Ollama timed out after {config.ollama_timeout}s.")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Ollama request failed: {e}")

    # estimate tokens using tiktoken if available; split prompt vs completion
    usage = {}
    try:
        import tiktoken
        mdl = model or "llama3.1"
        try:
            enc = tiktoken.encoding_for_model(mdl)
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        prompt_text = "\n".join([m.get("content", "") for m in messages])
        p_toks = len(enc.encode(prompt_text))
        c_toks = len(enc.encode(text))
        usage = {"prompt_tokens": int(p_toks), "completion_tokens": int(c_toks), "total_tokens": int(p_toks + c_toks)}
    except Exception:
        import re
        p_words = len(re.findall(r"\w+", "\n".join([m.get("content", "") for m in messages])))
        c_words = len(re.findall(r"\w+", text))
        usage = {"prompt_tokens": int(p_words), "completion_tokens": int(c_words), "total_tokens": int(p_words + c_words)}
    return text, usage

# ----------------------------- prompt builder -----------------------------
def _build_prompt(question: str, contexts: list[dict], mode: str) -> list[dict]:
    if mode == "strict":
        sys = (
            "Answer the user's question using ONLY the numbered context below. "
            "Cite every claim with [n] referring to the context number. "
            "If the answer is not in the context, reply exactly: I don't know based on these files."
        )
    elif mode == "fallback":
        sys = (
            "Prefer the numbered context to answer. Cite context with [n]. "
            "If the context is insufficient, you may use general knowledge — but say so explicitly with the phrase '(general knowledge)'."
        )
    else:  # pure_llm
        sys = "Answer using your general knowledge. Be precise and concise."

    if contexts:
        ctx_text = "\n\n".join(
            f"[{i+1}] {_cite_label(c)}\n{c['text']}" for i, c in enumerate(contexts)
        )
        user = f"Context:\n{ctx_text}\n\nQuestion: {question}"
    else:
        user = f"Question: {question}"
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]


def _cite_label(c: dict) -> str:
    m = c.get("meta", {}) or {}
    parts = [m.get("rel_path") or m.get("source", "?")]
    if m.get("page"):    parts.append(f"p.{m['page']}")
    if m.get("slide"):   parts.append(f"slide {m['slide']}")
    if m.get("sheet"):   parts.append(f"sheet {m['sheet']}")
    if m.get("section"): parts.append(f"\"{m['section']}\"")
    return " · ".join(parts)


# ----------------------------- main entrypoint -----------------------------
def answer_question(
    question: str,
    folder_path: str,
    chroma_dir: str,
    top_k: int,
    backend: Literal["OpenAI", "Ollama"],
    config: AppConfig,
    ollama_model: str | None = None,
    openai_model: str | None = None,
    retrieval_mode: Literal["strict", "fallback", "pure_llm"] = "fallback",
):
    if retrieval_mode == "pure_llm":
        msgs = _build_prompt(question, [], mode="pure_llm")
        if backend == "OpenAI":
            text, usage = call_openai(msgs, config, model=openai_model)
        else:
            text, usage = call_ollama(msgs, config, ollama_model)
        return {"answer": text, "sources": [], "debug": {}, "tokens": usage}

    client = chromadb.PersistentClient(path=chroma_dir)
    collection = client.get_or_create_collection(name="docs")
    embedder = _embedder(config.embedding_model)
    reranker = _reranker(config.reranker_model)

    # 1. query expansion (skip for very short keyword queries)
    if config.use_query_expansion and len(question.split()) >= 4:
        queries = _expand_query(question, backend, config, ollama_model, openai_model=openai_model)
    else:
        queries = [question]

    # 2. hybrid retrieve per query, fuse
    all_ranked = []
    for q in queries:
        v = _vector_search(q, collection, embedder, config.initial_k)
        b = _bm25_search(q, collection, config.initial_k)
        all_ranked.extend([v, b])
    fused = _rrf_fuse(*all_ranked)[: config.rerank_k]

    # 3. cross-encoder rerank
    reranked = _rerank(question, fused, reranker, top=max(top_k * 2, top_k))

    # 4. fallback decision based on reranker score
    max_rr = reranked[0]["rerank_score"] if reranked else -1e9
    if retrieval_mode == "fallback" and max_rr < config.fallback_threshold:
        contexts = []
    else:
        # 5. MMR over child text first — small, semantically tight, cheap to embed.
        child_pool = reranked[: max(top_k * 2, top_k)]
        if config.use_mmr and len(child_pool) > 1:
            qvec = embedder.encode([question], normalize_embeddings=True)[0]
            child_pool = _mmr(qvec, child_pool, embedder, k=top_k, lam=config.mmr_lambda)
        else:
            child_pool = child_pool[:top_k]

        # 6. parent-child expansion: swap each child for its parent window
        # (concatenated sibling chunks). Citation metadata stays anchored on
        # the matched child so source pins remain precise.
        if getattr(config, "use_parent_child", True):
            contexts = _expand_to_parents(child_pool, collection)[:top_k]
        else:
            contexts = child_pool[:top_k]

    # 7. LLM
    msgs = _build_prompt(question, contexts, mode=retrieval_mode)
    if backend == "OpenAI":
        text, usage = call_openai(msgs, config, model=openai_model)
    else:
        text, usage = call_ollama(msgs, config, ollama_model)

    sources = [
        {
            "n": i + 1,
            "source": (c["meta"] or {}).get("rel_path") or (c["meta"] or {}).get("source", ""),
            "page": (c["meta"] or {}).get("page"),
            "slide": (c["meta"] or {}).get("slide"),
            "sheet": (c["meta"] or {}).get("sheet"),
            "section": (c["meta"] or {}).get("section"),
            "score": float(c.get("rerank_score", 0.0)),
            "snippet": (c.get("child_text") or c["text"])[:280],
            "parent_chunks": c.get("children", []),
        }
        for i, c in enumerate(contexts)
    ]

    return {
        "answer": text,
        "sources": sources,
        "debug": {
            "queries": queries,
            "fused_count": len(fused),
            "reranked_count": len(reranked),
            "max_rerank_score": max_rr,
            "parent_child": bool(getattr(config, "use_parent_child", True)),
        },
        "tokens": usage,
    }
