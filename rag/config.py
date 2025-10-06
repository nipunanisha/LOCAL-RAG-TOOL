import os
from dataclasses import dataclass


@dataclass
class AppConfig:
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    ollama_timeout: int = int(os.getenv("OLLAMA_TIMEOUT", "300"))


def get_chroma_dir_for_folder(folder_path: str) -> str:
    # Store per-folder index under hidden .chroma inside the folder
    return os.path.join(folder_path, ".chroma")

