"""Build and query a small Chroma vector index over chunked airline delay CSV rows."""

from __future__ import annotations

import csv
import os
import threading
from pathlib import Path

import tiktoken

_lock = threading.Lock()

# Bump when chunking / embedding strategy changes (invalidates Chroma + mtime guard).
_RAG_INDEX_VERSION = "3"

# OpenAI embedding models allow at most 8191 tokens per input; stay under that.
_MAX_EMBED_TOKENS = 8000

# Chroma calls OpenAI with the whole list passed to `embeddings.create`; keep batches small.
_EMBED_ADD_BATCH_SIZE = 64


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _truncate_for_embedding(text: str) -> str:
    enc = tiktoken.get_encoding("cl100k_base")
    ids = enc.encode(text)
    if len(ids) <= _MAX_EMBED_TOKENS:
        return text
    return enc.decode(ids[:_MAX_EMBED_TOKENS])


def _indexed_mtime_path(persist_dir: Path) -> Path:
    return persist_dir / "indexed_csv_mtime.txt"


def _index_version_path(persist_dir: Path) -> Path:
    return persist_dir / "rag_index_version.txt"


def _chunk_csv(path: Path, rows_per_chunk: int = 15) -> list[str]:
    """Split CSV into overlapping-ish text chunks (header context repeated per chunk)."""
    chunks: list[str] = []
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return []
        header_line = ",".join(header)
        batch: list[str] = []
        for row in reader:
            batch.append(",".join(row))
            if len(batch) >= rows_per_chunk:
                chunks.append(_format_chunk(header_line, batch))
                batch = []
        if batch:
            chunks.append(_format_chunk(header_line, batch))
    return chunks


def _format_chunk(header_line: str, lines: list[str]) -> str:
    body = "\n".join(lines)
    return (
        "Airline delay cause reporting (comma-separated columns). "
        f"Header: {header_line}\nRows:\n{body}"
    )


def get_or_create_collection(csv_path: Path):
    """Return a Chroma collection embedded with OpenAI; rebuild if CSV changes."""
    import chromadb
    from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is required for RAG embeddings on the MCP server. "
            "The LangGraph app forwards this into the MCP subprocess env."
        )

    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    persist_dir = _project_root() / ".chroma_airline_reliability"
    persist_dir.mkdir(parents=True, exist_ok=True)

    mtime = csv_path.stat().st_mtime
    meta = _indexed_mtime_path(persist_dir)
    ver = _index_version_path(persist_dir)

    ef = OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name=os.environ.get("AIRLINE_DELAY_EMBED_MODEL", "text-embedding-3-small"),
    )
    client = chromadb.PersistentClient(path=str(persist_dir))
    name = "airline_reliability_chunks"

    with _lock:
        version_ok = ver.is_file() and ver.read_text().strip() == _RAG_INDEX_VERSION
        if not version_ok:
            rebuild = True
        elif not meta.is_file():
            rebuild = True
        else:
            try:
                rebuild = float(meta.read_text().strip()) != mtime
            except ValueError:
                rebuild = True

        coll = None
        if not rebuild:
            try:
                coll = client.get_collection(name=name, embedding_function=ef)
                if coll.count() == 0:
                    rebuild = True
            except Exception:
                rebuild = True

        if rebuild:
            try:
                client.delete_collection(name)
            except Exception:
                pass
            coll = client.create_collection(name=name, embedding_function=ef)
            rows_per = int(os.environ.get("AIRLINE_DELAY_RAG_ROWS_PER_CHUNK", "15"))
            chunks = _chunk_csv(csv_path, rows_per_chunk=max(3, min(rows_per, 50)))
            if not chunks:
                raise ValueError(f"No rows to index in {csv_path}")
            docs = []
            for raw in chunks:
                t = _truncate_for_embedding(raw).strip()
                if t:
                    docs.append(t)
            if not docs:
                raise ValueError("All chunks were empty after truncation.")
            ids = [f"c{i}" for i in range(len(docs))]
            for start in range(0, len(docs), _EMBED_ADD_BATCH_SIZE):
                batch_docs = docs[start : start + _EMBED_ADD_BATCH_SIZE]
                batch_ids = ids[start : start + _EMBED_ADD_BATCH_SIZE]
                coll.add(documents=batch_docs, ids=batch_ids)
            meta.write_text(str(mtime))
            ver.write_text(_RAG_INDEX_VERSION)

        if coll is None:
            coll = client.get_collection(name=name, embedding_function=ef)

    return coll


def search_chunks(csv_path: Path, query: str, n_results: int) -> str:
    q = query.strip()
    if not q:
        return "ERROR: empty search query."
    q = _truncate_for_embedding(q)
    coll = get_or_create_collection(csv_path)
    res = coll.query(query_texts=[q], n_results=n_results)
    docs = (res.get("documents") or [[]])[0]
    if not docs:
        return "No matching chunks returned."
    parts = []
    for i, doc in enumerate(docs):
        parts.append(f"--- match {i + 1} ---\n{doc}")
    return "\n\n".join(parts)
