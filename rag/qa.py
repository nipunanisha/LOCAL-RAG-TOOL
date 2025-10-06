from typing import Literal
import requests
import chromadb
from rag.config import AppConfig

def retrieve(query: str, chroma_dir: str, top_k: int = 5):
    """Retrieve top-k documents from Chroma DB."""
    client = chromadb.PersistentClient(path=chroma_dir)
    collection = client.get_or_create_collection(name="docs")
    res = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["distances", "documents", "metadatas"]
    )
    docs = []
    for i in range(len(res["ids"][0])):
        docs.append({
            "text": res["documents"][0][i],
            "source": res["metadatas"][0][i]["source"],
            "score": float(res["distances"][0][i])
        })
    return docs


def call_openai(messages: list[dict], config: AppConfig) -> str:
    """Call OpenAI GPT-4o-mini using the latest SDK with custom http client."""
    from openai import OpenAI, DefaultHttpxClient
    import httpx

    if not config.openai_api_key:
        raise ValueError("OpenAI API key not set in AppConfig!")

    http_client = DefaultHttpxClient(transport=httpx.HTTPTransport())
    client = OpenAI(api_key=config.openai_api_key, http_client=http_client)

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        temperature=0.2,
    )
    return resp.choices[0].message.content or ""


def call_ollama(messages: list[dict], config: AppConfig, model: str | None) -> str:
    """Call Ollama API."""
    url = f"{config.ollama_base_url}/api/chat"
    model_name = model or "llama3.1"
    payload = {"model": model_name, "messages": messages, "stream": False, "options": {"temperature": 0.2}}

    try:
        r = requests.post(url, json=payload, timeout=config.ollama_timeout)
        r.raise_for_status()
        j = r.json()
        return j.get("message", {}).get("content", "")
    except requests.exceptions.ReadTimeout:
        raise RuntimeError(f"Ollama timed out after {config.ollama_timeout}s.")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Ollama request failed: {e}")


def build_prompt(question: str, contexts: list[dict], mode: Literal["strict","fallback","pure_llm"]) -> list[dict]:
    """Build system + user prompt for LLM."""
    if mode == "strict":
        system = "Answer using ONLY the provided context. Say 'I don't know' if not present."
    elif mode == "fallback":
        system = "Use context if relevant, otherwise answer using your general knowledge."
    else:  # pure_llm
        system = "Answer the question using your general knowledge. Be clear and concise."

    context_text = "\n\n".join(f"Source: {c['source']}\n{c['text']}" for c in contexts)
    user = f"Question: {question}\n\nContext:\n{context_text}" if contexts else f"Question: {question}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def answer_question(
    question: str,
    folder_path: str,
    chroma_dir: str,
    top_k: int,
    backend: Literal["OpenAI", "Ollama"],
    config: AppConfig,
    ollama_model: str | None = None,
    retrieval_mode: Literal["strict","fallback","pure_llm"] = "strict"
):
    """Retrieve context and answer question using chosen LLM backend."""
    
    if retrieval_mode == "pure_llm":
        contexts = []
    else:
        contexts = retrieve(question, chroma_dir, top_k=top_k)
        if retrieval_mode == "fallback" and (not contexts or all(c["score"] > 1.5 for c in contexts)):
            contexts = []

    messages = build_prompt(question, contexts, mode=retrieval_mode)

    if backend == "OpenAI":
        answer = call_openai(messages, config)
    else:
        answer = call_ollama(messages, config, model=ollama_model)

    return {"answer": answer, "sources": contexts}
