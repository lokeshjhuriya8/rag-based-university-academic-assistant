"""RAG utilities with incremental indexing for local Ollama or cloud Gemini."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
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
OLLAMA_COLLECTION_NAME = "university_documents"  # Preserves the existing local index.
GEMINI_COLLECTION_NAME = "university_documents_gemini"
LOGGER = logging.getLogger(__name__)

# A single worker and write lock prevent unsafe concurrent Chroma writes. Reads
# intentionally take no write lock, so questions remain available while indexing.
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


def _setting(name: str, default: str | None = None) -> str | None:
    """Read .env locally and Streamlit Cloud Secrets when deployed."""
    value = os.getenv(name)
    if value:
        return value
    try:
        import streamlit as st

        secret = st.secrets.get(name)
        return str(secret) if secret is not None else default
    except Exception:
        return default


def config() -> AppConfig:
    provider = (_setting("LLM_PROVIDER", "ollama") or "ollama").lower()
    if provider not in {"ollama", "gemini"}:
        raise ValueError("Unsupported LLM provider. Use 'ollama' or 'gemini'.")
    is_ollama = provider == "ollama"
    api_key = "ollama" if is_ollama else _setting("GEMINI_API_KEY")
    if not is_ollama and not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing. Add it to Streamlit Cloud Secrets.")
    embedding_model = _setting("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text") if is_ollama else _setting("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
    if not is_ollama and embedding_model != "gemini-embedding-001":
        raise ValueError("GEMINI_EMBEDDING_MODEL must be 'gemini-embedding-001'.")
    return AppConfig(
        provider=provider,
        api_key=api_key,
        base_url=_setting("OLLAMA_BASE_URL", "http://localhost:11434/v1") if is_ollama else None,
        chat_model=_setting("OLLAMA_CHAT_MODEL", "llama3.2") if is_ollama else _setting("GEMINI_CHAT_MODEL", "gemini-3.5-flash-lite"),
        embedding_model=embedding_model,
        top_k=int(_setting("TOP_K", "3") or "3"),
        chunk_size=int(_setting("CHUNK_SIZE", "1200") or "1200"),
        chunk_overlap=int(_setting("CHUNK_OVERLAP", "100") or "100"),
    )


def ollama_client() -> OpenAI:
    cfg = config()
    if cfg.provider != "ollama":
        raise RuntimeError("Ollama client requested while Ollama is not selected.")
    return OpenAI(api_key="ollama", base_url=cfg.base_url)


def gemini_client():
    cfg = config()
    if cfg.provider != "gemini":
        raise RuntimeError("Gemini client requested while Gemini is not selected.")
    try:
        from google import genai
    except ImportError as error:
        raise RuntimeError("Gemini support is not installed. Run: pip install -r requirements.txt") from error
    return genai.Client(api_key=cfg.api_key)


def collection():
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    cfg = config()
    # The embedding model becomes part of Gemini's collection identity. This
    # prevents a model/dimension change from silently mixing incompatible vectors.
    if cfg.provider == "ollama":
        name = OLLAMA_COLLECTION_NAME
    else:
        model_tag = hashlib.sha256(cfg.embedding_model.encode()).hexdigest()[:10]
        name = f"{GEMINI_COLLECTION_NAME}_{model_tag}"
    db = chromadb.PersistentClient(path=str(STORAGE_DIR))
    return db.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})


def _manifest_path() -> Path:
    # Keep the original Ollama manifest. Gemini needs a separate manifest because
    # the two providers generate vectors with different dimensions.
    cfg = config()
    if cfg.provider == "ollama":
        return ROOT / "storage" / "index_manifest.json"
    model_tag = hashlib.sha256(cfg.embedding_model.encode()).hexdigest()[:10]
    return ROOT / "storage" / f"index_manifest_gemini_{model_tag}.json"


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


def _safe_gemini_error(error: Exception) -> str:
    """Return a useful API error without ever echoing a credential."""
    message = str(error).replace("\n", " ")
    message = re.sub(r"(?i)(api[_ -]?key\s*[=:]\s*)[^\s,;]+", r"\1[redacted]", message)
    message = re.sub(r"(?i)(key=)[^&\s]+", r"\1[redacted]", message)
    return message[:700] or error.__class__.__name__


def _gemini_vector(embedding: object) -> list[float]:
    """Support the official SDK's .values field and compatible SDK variants."""
    values = getattr(embedding, "values", None)
    if values is None:
        nested = getattr(embedding, "embedding", None)
        values = getattr(nested, "values", nested)
    if values is None:
        raise RuntimeError("Gemini returned an embedding without vector values.")
    return list(values)


def embed(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    """Embed text through the selected provider without duplicating RAG logic."""
    cfg = config()
    client = None
    try:
        if cfg.provider == "ollama":
            response = ollama_client().embeddings.create(model=cfg.embedding_model, input=texts)
            return [item.embedding for item in response.data]
        # gemini-embedding-001 uses the documented embed_content interface.
        # The same model and returned vector dimension are used for documents
        # and queries; task_type is intentionally not sent here.
        # Keep a named strong reference until the synchronous request completes.
        # Client.close() is performed only in finally, after success or failure.
        client = gemini_client()
        response = client.models.embed_content(
            model=cfg.embedding_model,
            contents=texts,
        )
        embeddings = list(response.embeddings or [])
        if len(embeddings) != len(texts):
            raise RuntimeError(f"Gemini returned {len(embeddings)} embeddings for {len(texts)} input texts.")
        vectors = [_gemini_vector(item) for item in embeddings]
        if not vectors or any(not vector for vector in vectors):
            raise RuntimeError("Gemini returned an empty embedding vector.")
        dimensions = len(vectors[0])
        if any(len(vector) != dimensions for vector in vectors):
            raise RuntimeError("Gemini returned inconsistent embedding dimensions.")
        return vectors
    except Exception as error:
        if cfg.provider == "ollama":
            raise RuntimeError("Ollama is not running or the configured model is unavailable. Start Ollama and try again.") from error
        message = _safe_gemini_error(error)
        LOGGER.warning("Gemini embedding request failed: %s", message)
        raise RuntimeError(f"Gemini embedding request failed: {message}") from error
    finally:
        if client is not None:
            try:
                client.close()
            except Exception as close_error:
                # Never replace an API/indexing error with a cleanup error.
                LOGGER.warning("Gemini client cleanup failed: %s", _safe_gemini_error(close_error))


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_manifest() -> dict[str, str]:
    try:
        return json.loads(_manifest_path().read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _write_manifest(manifest: dict[str, str]) -> None:
    path = _manifest_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _set_job(filename: str, **changes: object) -> None:
    with _STATE_LOCK:
        _JOBS.setdefault(filename, {"filename": filename}).update(changes)


def indexing_snapshot() -> list[dict[str, object]]:
    with _STATE_LOCK:
        return [dict(job) for job in _JOBS.values()]


def indexing_in_progress() -> bool:
    return any(job.get("state") in {"queued", "indexing"} for job in indexing_snapshot())


def _ids_for(target, where: dict[str, str]) -> list[str]:
    return list(target.get(where=where, include=[]).get("ids", []))


def _index_one_file(path: Path) -> None:
    """Background worker task. It never touches Streamlit session state."""
    filename = path.name
    try:
        _set_job(filename, state="indexing", message="Calculating document fingerprint…", indexed_chunks=0)
        current_hash = file_hash(path)
        _set_job(filename, file_hash=current_hash)
        target = collection()

        with _WRITE_LOCK:
            manifest = _read_manifest()
            same_hash_ids = _ids_for(target, {"file_hash": current_hash})
            existing_source_ids = _ids_for(target, {"source": filename})
            if same_hash_ids or (manifest.get(filename) == current_hash and existing_source_ids):
                manifest[filename] = current_hash
                _write_manifest(manifest)
                _set_job(filename, state="complete", message="Already indexed; skipped.", indexed_chunks=0)
                return
            if filename not in manifest and existing_source_ids and config().provider == "ollama":
                # Adopt the pre-Gemini local index once without re-embedding it.
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
                records.append((chunk_id, text_chunk, {"source": filename, "file_hash": current_hash, "embedding_model": settings.embedding_model, "page": page, "chunk": chunk_number}))
        if not records:
            raise RuntimeError("No extractable text was found. Use a text-based PDF or run OCR first.")

        # Finish embedding before changing the collection. A failed embedding
        # therefore preserves old chunks for a modified document.
        batches: list[tuple[list[tuple[str, str, dict[str, object]]], list[list[float]]]] = []
        batch_size = 64
        for offset in range(0, len(records), batch_size):
            batch = records[offset : offset + batch_size]
            _set_job(filename, state="indexing", message=f"Embedding passages {offset + 1}-{min(offset + len(batch), len(records))} of {len(records)}…", indexed_chunks=offset)
            batches.append((batch, embed([record[1] for record in batch], "RETRIEVAL_DOCUMENT")))

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
    """Queue only new/changed PDFs and return immediately to Streamlit."""
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
            if prior.get("state") in {"queued", "indexing"} or (prior.get("state") == "failed" and prior.get("file_hash") == current_hash):
                continue
            _JOBS[filename] = {"filename": filename, "file_hash": current_hash, "state": "queued", "message": "Waiting to be indexed…", "indexed_chunks": 0}
        _INDEX_EXECUTOR.submit(_index_one_file, path)
        queued += 1
    return queued


def retrieve(question: str) -> list[dict[str, object]]:
    target = collection()
    count = target.count()
    if count == 0:
        raise RuntimeError("The knowledge base is empty. Add a PDF and wait for indexing to finish.")
    result = target.query(query_embeddings=embed([question], "RETRIEVAL_QUERY"), n_results=min(config().top_k, count), include=["documents", "metadatas", "distances"])
    return [{"text": text, "source": metadata["source"], "page": metadata["page"], "distance": distance} for text, metadata, distance in zip(result["documents"][0], result["metadatas"][0], result["distances"][0])]


def answer(question: str, history: list[dict[str, str]] | None = None) -> tuple[str, list[dict[str, object]]]:
    sources = retrieve(question)
    context = "\n\n".join(f"[Source {number}: {source['source']}, page {source['page']}]\n{source['text']}" for number, source in enumerate(sources, start=1))
    system = """You are a careful university academic assistant. Answer only from the supplied university-document context. Do not invent policies, deadlines, or facts. If the context does not answer the question, say exactly that you could not find it in the indexed university documents. Cite every factual claim with source markers such as [Source 1]. Be concise and helpful."""
    final_question = f"University-document context:\n{context}\n\nQuestion: {question}"
    cfg = config()
    client = None
    try:
        if cfg.provider == "ollama":
            messages = [{"role": "system", "content": system}]
            for item in (history or [])[-6:]:
                messages.append({"role": item["role"], "content": item["content"]})
            messages.append({"role": "user", "content": final_question})
            response = ollama_client().chat.completions.create(model=cfg.chat_model, messages=messages, temperature=0.1)
            text = response.choices[0].message.content
        else:
            from google.genai import types

            contents = []
            for item in (history or [])[-6:]:
                role = "model" if item["role"] == "assistant" else "user"
                contents.append(types.Content(role=role, parts=[types.Part.from_text(text=item["content"])]))
            contents.append(types.Content(role="user", parts=[types.Part.from_text(text=final_question)]))
            # Keep the named client alive for the whole synchronous request.
            client = gemini_client()
            response = client.models.generate_content(model=cfg.chat_model, contents=contents, config=types.GenerateContentConfig(system_instruction=system, temperature=0.1))
            text = response.text
    except Exception as error:
        if cfg.provider == "ollama":
            raise RuntimeError("Ollama is not running or the configured model is unavailable. Start Ollama and try again.") from error
        message = _safe_gemini_error(error)
        LOGGER.warning("Gemini answer request failed: %s", message)
        raise RuntimeError(f"Gemini answer request failed: {message}") from error
    finally:
        if client is not None:
            try:
                client.close()
            except Exception as close_error:
                LOGGER.warning("Gemini client cleanup failed: %s", _safe_gemini_error(close_error))
    return text or "I could not generate an answer.", sources
