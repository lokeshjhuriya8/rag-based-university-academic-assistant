"""RAG utilities plus safe, incremental background PDF indexing."""
from __future__ import annotations

import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

load_dotenv()

ROOT = Path(__file__).parent
DOCUMENTS_DIR = ROOT / "data" / "documents"
STORAGE_DIR = ROOT / "storage" / "chroma"
MANIFEST_PATH = ROOT / "storage" / "index_manifest.json"
COLLECTION_NAME = "university_documents"

# One process may only write to Chroma at a time. Reads deliberately do not use
# this lock, so retrieval remains available while a PDF is being embedded.
_WRITE_LOCK = threading.Lock()
_STATE_LOCK = threading.Lock()
_INDEX_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pdf-indexer")
_JOBS: dict[str, dict[str, object]] = {}


@dataclass(frozen=True)
class AppConfig:
    provider: str
    api_key: str | None
    base_url: str | None
    chat_model: str
    embedding_model: str
    top_k: int
    chunk_size: int
    chunk_overlap: int


def config() -> AppConfig:
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    if provider not in {"openai", "ollama"}:
        raise ValueError("LLM_PROVIDER must be 'openai' or 'ollama'.")
    ollama = provider == "ollama"
    return AppConfig(
        provider=provider,
        api_key="ollama" if ollama else os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1") if ollama else None,
        chat_model=os.getenv("OLLAMA_CHAT_MODEL", "llama3.2") if ollama else os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        embedding_model=os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text") if ollama else os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        top_k=int(os.getenv("TOP_K", "3")),
        chunk_size=int(os.getenv("CHUNK_SIZE", "1200")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "100")),
    )


def client() -> OpenAI:
    cfg = config()
    if cfg.provider == "openai" and not cfg.api_key:
        raise RuntimeError("OPENAI_API_KEY is missing. Add it to your .env file.")
    return OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)


def collection():
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    db = chromadb.PersistentClient(path=str(STORAGE_DIR))
    return db.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})


def split_text(text: str, size: int, overlap: int) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start + size // 2:
                end = boundary
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def read_pdf_pages(path: Path) -> Iterable[tuple[int, str]]:
    reader = PdfReader(str(path))
    for page_number, page in enumerate(reader.pages, start=1):
        yield page_number, page.extract_text() or ""


def embed(texts: list[str]) -> list[list[float]]:
    response = client().embeddings.create(model=config().embedding_model, input=texts)
    return [item.embedding for item in response.data]


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_manifest() -> dict[str, str]:
    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _write_manifest(manifest: dict[str, str]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = MANIFEST_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(MANIFEST_PATH)


def _set_job(filename: str, **changes: object) -> None:
    with _STATE_LOCK:
        current = _JOBS.setdefault(filename, {"filename": filename})
        current.update(changes)


def indexing_snapshot() -> list[dict[str, object]]:
    """Thread-safe job state for the Streamlit thread. No session state is used."""
    with _STATE_LOCK:
        return [dict(job) for job in _JOBS.values()]


def indexing_in_progress() -> bool:
    return any(job.get("state") in {"queued", "indexing"} for job in indexing_snapshot())


def _ids_for(target, where: dict[str, str]) -> list[str]:
    result = target.get(where=where, include=[])
    return list(result.get("ids", []))


def _index_one_file(path: Path) -> None:
    filename = path.name
    try:
        _set_job(filename, state="indexing", message="Calculating document fingerprint…", indexed_chunks=0)
        current_hash = file_hash(path)
        _set_job(filename, file_hash=current_hash)
        target = collection()

        # A manifest makes repeated app starts cheap. For a pre-existing index
        # created by older app versions, a matching filename is adopted once
        # rather than re-embedded.
        with _WRITE_LOCK:
            manifest = _read_manifest()
            same_hash_ids = _ids_for(target, {"file_hash": current_hash})
            existing_source_ids = _ids_for(target, {"source": filename})
            if same_hash_ids:
                manifest[filename] = current_hash
                _write_manifest(manifest)
                _set_job(filename, state="complete", message="Already indexed; skipped.", indexed_chunks=0)
                return
            if manifest.get(filename) == current_hash and existing_source_ids:
                _set_job(filename, state="complete", message="Already indexed; skipped.", indexed_chunks=0)
                return
            if filename not in manifest and existing_source_ids:
                manifest[filename] = current_hash
                _write_manifest(manifest)
                _set_job(filename, state="complete", message="Existing index adopted; skipped.", indexed_chunks=0)
                return

        _set_job(filename, state="indexing", message="Extracting pages…", indexed_chunks=0)
        records: list[tuple[str, str, dict[str, object]]] = []
        settings = config()
        for page, text in read_pdf_pages(path):
            for chunk_number, text_chunk in enumerate(split_text(text, settings.chunk_size, settings.chunk_overlap)):
                chunk_id = hashlib.sha256(f"{current_hash}:{page}:{chunk_number}:{text_chunk}".encode()).hexdigest()
                records.append((chunk_id, text_chunk, {"source": filename, "file_hash": current_hash, "page": page, "chunk": chunk_number}))
        if not records:
            raise RuntimeError("No extractable text was found. Use a text-based PDF or run OCR first.")

        # Embed first. If this fails, no existing Chroma records are changed.
        batches: list[tuple[list[tuple[str, str, dict[str, object]]], list[list[float]]]] = []
        batch_size = 64
        for offset in range(0, len(records), batch_size):
            batch = records[offset : offset + batch_size]
            _set_job(filename, state="indexing", message=f"Embedding passages {offset + 1}-{min(offset + len(batch), len(records))} of {len(records)}…", indexed_chunks=offset)
            batches.append((batch, embed([record[1] for record in batch])))

        # The short write section is serialized. New records are committed before
        # old same-filename records are removed, so a failed update preserves the
        # prior document and never affects unrelated PDFs.
        _set_job(filename, state="indexing", message="Saving to knowledge base…", indexed_chunks=0)
        with _WRITE_LOCK:
            target = collection()
            old_ids = _ids_for(target, {"source": filename})
            new_ids = {record[0] for batch, _ in batches for record in batch}
            completed = 0
            for batch, vectors in batches:
                target.upsert(ids=[record[0] for record in batch], documents=[record[1] for record in batch], metadatas=[record[2] for record in batch], embeddings=vectors)
                completed += len(batch)
                _set_job(filename, state="indexing", message=f"Saving passages {completed} of {len(records)}…", indexed_chunks=completed)
            stale_ids = [item_id for item_id in old_ids if item_id not in new_ids]
            if stale_ids:
                target.delete(ids=stale_ids)
            manifest = _read_manifest()
            manifest[filename] = current_hash
            _write_manifest(manifest)

        _set_job(filename, state="complete", message="Added to the knowledge base.", indexed_chunks=len(records))
    except Exception as error:
        _set_job(filename, state="failed", message=str(error), indexed_chunks=0)


def enqueue_unindexed_documents(paths: Iterable[Path]) -> int:
    """Queue files for background incremental indexing; returns newly queued count."""
    queued = 0
    for path in paths:
        if path.suffix.lower() != ".pdf" or not path.exists():
            continue
        filename = path.name
        try:
            current_hash = file_hash(path)
        except OSError:
            continue
        with _WRITE_LOCK:
            if _read_manifest().get(filename) == current_hash:
                continue
        with _STATE_LOCK:
            prior = _JOBS.get(filename, {})
            state = prior.get("state")
            if state in {"queued", "indexing"} or (state == "failed" and prior.get("file_hash") == current_hash):
                continue
            _JOBS[filename] = {"filename": filename, "file_hash": current_hash, "state": "queued", "message": "Waiting to be indexed…", "indexed_chunks": 0}
        _INDEX_EXECUTOR.submit(_index_one_file, path)
        queued += 1
    return queued


def index_documents(rebuild: bool = False) -> tuple[int, int]:
    """Compatibility helper: incrementally indexes only new/changed PDFs.

    ``rebuild`` is intentionally ignored: normal application use never deletes
    the Chroma collection or re-embeds unchanged files.
    """
    files = sorted(DOCUMENTS_DIR.glob("*.pdf"))
    if not files:
        raise RuntimeError("No PDFs found. Put university documents in data/documents/.")
    before = collection().count()
    for path in files:
        _index_one_file(path)
    return len(files), max(0, collection().count() - before)


def retrieve(question: str) -> list[dict[str, object]]:
    # No write lock here: queries can continue while the background worker is
    # embedding or committing an unrelated document.
    target = collection()
    count = target.count()
    if count == 0:
        raise RuntimeError("The knowledge base is empty. Add a PDF and wait for indexing to finish.")
    result = target.query(query_embeddings=embed([question]), n_results=min(config().top_k, count), include=["documents", "metadatas", "distances"])
    return [{"text": text, "source": metadata["source"], "page": metadata["page"], "distance": distance} for text, metadata, distance in zip(result["documents"][0], result["metadatas"][0], result["distances"][0])]


def answer(question: str, history: list[dict[str, str]] | None = None) -> tuple[str, list[dict[str, object]]]:
    sources = retrieve(question)
    context = "\n\n".join(f"[Source {number}: {source['source']}, page {source['page']}]\n{source['text']}" for number, source in enumerate(sources, start=1))
    system = """You are a careful university academic assistant. Answer only from the supplied university-document context. Do not invent policies, deadlines, or facts. If the context does not answer the question, say exactly that you could not find it in the indexed university documents. Cite every factual claim with source markers such as [Source 1]. Be concise and helpful."""
    messages = [{"role": "system", "content": system}]
    for item in (history or [])[-6:]:
        messages.append({"role": item["role"], "content": item["content"]})
    messages.append({"role": "user", "content": f"University-document context:\n{context}\n\nQuestion: {question}"})
    response = client().chat.completions.create(model=config().chat_model, messages=messages, temperature=0.1)
    return response.choices[0].message.content or "I could not generate an answer.", sources
