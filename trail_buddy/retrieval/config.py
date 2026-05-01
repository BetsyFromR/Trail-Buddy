from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAG_STORE = PROJECT_ROOT / "rag_store"
DEFAULT_COLLECTION_NAME = "trail_buddy_knowledge"


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


def _default_collection_name(rag_store_dir: Path) -> str:
    names = available_collection_names(rag_store_dir)
    if names:
        return names[0]
    return DEFAULT_COLLECTION_NAME


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

    rag_store_dir: Path = field(
        default_factory=lambda: _path_env(
            "TRAIL_BUDDY_RAG_STORE_DIR",
            DEFAULT_RAG_STORE,
        )
    )
    collection_name: str | None = field(
        default_factory=lambda: os.getenv("TRAIL_BUDDY_RAG_COLLECTION") or None
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "TRAIL_BUDDY_RAG_EMBEDDING_MODEL",
            "all-MiniLM-L6-v2",
        )
    )
    retriever_k: int = field(
        default_factory=lambda: _int_env("TRAIL_BUDDY_RAG_RETRIEVER_K", 5)
    )

    def __post_init__(self) -> None:
        if self.collection_name is None:
            object.__setattr__(
                self,
                "collection_name",
                _default_collection_name(self.rag_store_dir),
            )

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
    return RetrievalSettings()


def ensure_rag_store_dirs(settings: RetrievalSettings | None = None) -> None:
    resolved = settings or get_retrieval_settings()
    resolved.raw_data_dir.mkdir(parents=True, exist_ok=True)
    resolved.processed_data_dir.mkdir(parents=True, exist_ok=True)
    resolved.chroma_dir.mkdir(parents=True, exist_ok=True)
