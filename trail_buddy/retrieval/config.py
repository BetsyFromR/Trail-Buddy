from __future__ import annotations

import os
import sqlite3
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAG_STORE = PROJECT_ROOT / "rag_store"
DEFAULT_RAG_CONFIG_FILE = PROJECT_ROOT / "rag_config.toml"


def _path_env(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    if not raw:
        return default
    path = Path(raw).expanduser()
    if not path.is_absolute():
        return PROJECT_ROOT / path
    return path


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _config_file_path() -> Path:
    return _path_env("TRAIL_BUDDY_RAG_CONFIG_FILE", DEFAULT_RAG_CONFIG_FILE)


def _load_rag_config() -> dict[str, Any]:
    path = _config_file_path()
    if not path.exists():
        return {}

    with path.open("rb") as file:
        data = tomllib.load(file)

    rag = data.get("rag", {})
    if not isinstance(rag, dict):
        return {}
    return rag


def _config_str(config: dict[str, Any], key: str, default: str | None = None) -> str | None:
    value = config.get(key, default)
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _config_int(config: dict[str, Any], key: str, default: int) -> int:
    value = config.get(key, default)
    if value is None or value == "":
        return default
    return int(value)


def _path_config(config: dict[str, Any], key: str, default: Path) -> Path:
    raw = config.get(key)
    if raw is None or str(raw).strip() == "":
        return default
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        return PROJECT_ROOT / path
    return path


def chroma_db_path(rag_store_dir: Path = DEFAULT_RAG_STORE) -> Path:
    return rag_store_dir / "indexes" / "chroma" / "chroma.sqlite3"


def available_collection_names(rag_store_dir: Path = DEFAULT_RAG_STORE) -> list[str]:
    db_path = chroma_db_path(rag_store_dir)
    if not db_path.exists():
        return []

    try:
        with sqlite3.connect(db_path) as connection:
            rows = connection.execute(
                "select name from collections order by name"
            ).fetchall()
    except sqlite3.Error:
        return []

    return [str(row[0]) for row in rows if row and row[0]]


def available_raw_document_paths(rag_store_dir: Path = DEFAULT_RAG_STORE) -> list[Path]:
    raw_data_dir = rag_store_dir / "data" / "raw"
    if not raw_data_dir.exists():
        return []
    return sorted(
        path
        for path in raw_data_dir.glob("*.jsonl")
        if path.is_file()
    )


@dataclass(frozen=True)
class RetrievalSettings:
    """Filesystem locations for the external RAG store.

    Scraping and index refreshes should write to this store. The Trail Buddy app
    only reads from it at query time.
    """

    rag_store_dir: Path = DEFAULT_RAG_STORE
    config_file: Path = DEFAULT_RAG_CONFIG_FILE
    collection_name: str | None = field(
        default_factory=lambda: os.getenv("TRAIL_BUDDY_RAG_COLLECTION") or None
    )
    embedding_model: str = "all-MiniLM-L6-v2"
    retriever_k: int = 5

    @property
    def data_dir(self) -> Path:
        return self.rag_store_dir / "data"

    @property
    def raw_data_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_data_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def indexes_dir(self) -> Path:
        return self.rag_store_dir / "indexes"

    @property
    def chroma_dir(self) -> Path:
        return self.indexes_dir / "chroma"

    @property
    def document_paths(self) -> list[Path]:
        return available_raw_document_paths(self.rag_store_dir)


def get_retrieval_settings() -> RetrievalSettings:
    config_file = _config_file_path()
    config = _load_rag_config()
    collection_name = (
        os.getenv("TRAIL_BUDDY_RAG_COLLECTION")
        or _config_str(config, "collection")
        or None
    )
    return RetrievalSettings(
        rag_store_dir=_path_env(
            "TRAIL_BUDDY_RAG_STORE_DIR",
            _path_config(config, "store_dir", DEFAULT_RAG_STORE),
        ),
        config_file=config_file,
        collection_name=collection_name,
        embedding_model=(
            os.getenv("TRAIL_BUDDY_RAG_EMBEDDING_MODEL")
            or _config_str(config, "embedding_model", "all-MiniLM-L6-v2")
            or "all-MiniLM-L6-v2"
        ),
        retriever_k=_int_env(
            "TRAIL_BUDDY_RAG_RETRIEVER_K",
            _config_int(config, "retriever_k", 5),
        ),
    )


def ensure_rag_store_dirs(settings: RetrievalSettings | None = None) -> None:
    resolved = settings or get_retrieval_settings()
    resolved.raw_data_dir.mkdir(parents=True, exist_ok=True)
    resolved.processed_data_dir.mkdir(parents=True, exist_ok=True)
    resolved.chroma_dir.mkdir(parents=True, exist_ok=True)
