"""
Settings persistence. ~/.folder_rag/settings.json.
"""
from __future__ import annotations
import os
import json
import stat
from pathlib import Path
from typing import Any

CONFIG_DIR = Path(os.path.expanduser("~")) / ".folder_rag"
SETTINGS_PATH = str(CONFIG_DIR / "settings.json")

DEFAULTS: dict[str, Any] = {
    "openai_api_key": "",
    "openai_model": "gpt-4.1-mini",         # one of the OPENAI_MODELS below, or any compatible string
    "ollama_base_url": "http://localhost:11434",
    "ollama_model": "qwen3:8b",
    "default_backend": "Ollama",            # or "OpenAI"
    "default_retrieval_mode": "fallback",   # strict | fallback | pure_llm
    "default_top_k": 5,
    "last_folder": "",
    "theme": "dark",                         # dark | light
}

# Curated list surfaced in the UI dropdown. Free-form text is also accepted
# (the UI shows an "other…" option that reveals a text input).
OPENAI_MODELS = [
    "gpt-4.1-mini",
    "gpt-4.1",
    "gpt-4o-mini",
    "gpt-4o",
    "o4-mini",
]


def load_settings() -> dict:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if not os.path.exists(SETTINGS_PATH):
        save_settings(DEFAULTS)
        return dict(DEFAULTS)
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return dict(DEFAULTS)
    merged = dict(DEFAULTS)
    merged.update({k: v for k, v in data.items() if k in DEFAULTS})
    return merged


def save_settings(data: dict) -> dict:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    merged = load_settings() if os.path.exists(SETTINGS_PATH) else dict(DEFAULTS)
    merged.update({k: v for k, v in data.items() if k in DEFAULTS})
    tmp = SETTINGS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
    os.replace(tmp, SETTINGS_PATH)
    # restrict perms on POSIX
    try:
        os.chmod(SETTINGS_PATH, stat.S_IRUSR | stat.S_IWUSR)
    except Exception:
        pass
    return merged


def mask_key(key: str) -> str:
    if not key:
        return ""
    if len(key) <= 8:
        return "•" * len(key)
    return key[:3] + "···" + key[-4:]
