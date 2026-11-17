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
# A deletion may occur while a worker is embedding a document. The worker checks
# this set before any database write, so it cannot re-add a removed document.
_CANCELLED_JOBS: set[tuple[str, str]] = set()


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


def _persistent_client():
    """Return the project's persistent Chroma client without changing a collection."""
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(STORAGE_DIR))


def _managed_collections() -> list[object]:
    """Return every project collection, including old provider-specific ones."""
    managed: list[object] = []
    for item in _persistent_client().list_collections():
        name = getattr(item, "name", item)
        if name == OLLAMA_COLLECTION_NAME or str(name).startswith(GEMINI_COLLECTION_NAME):
            managed.append(item if hasattr(item, "get") else _persistent_client().get_collection(name=str(name)))
    return managed


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


def _all_manifest_paths() -> list[Path]:
    """Find manifests for local Ollama and every Gemini embedding collection."""
    storage_root = ROOT / "storage"
    return [storage_root / "index_manifest.json", *storage_root.glob("index_manifest_gemini_*.json")]


def safe_document_path(filename: str) -> Path:
    """Resolve a PDF name only inside the managed document directory."""
    if not filename or Path(filename).name != filename or Path(filename).suffix.lower() != ".pdf":
        raise ValueError("Invalid document name.")
    documents_root = DOCUMENTS_DIR.resolve()
    candidate = (documents_root / filename).resolve()
    try:
        candidate.relative_to(documents_root)
    except ValueError as error:
        raise ValueError("Document path must stay inside data/documents.") from error
    return candidate


def _cancelled(filename: str, document_hash: str) -> bool:
    with _STATE_LOCK:
        return (filename, document_hash) in _CANCELLED_JOBS


def _metadata_records(target) -> list[tuple[str, dict[str, object], str]]:
    """Read chunk data once and normalise Chroma's parallel result lists."""
    result = target.get(include=["documents", "metadatas"])
    return [
        (str(chunk_id), metadata or {}, document or "")
        for chunk_id, metadata, document in zip(
            result.get("ids", []), result.get("metadatas", []), result.get("documents", [])
        )
    ]


def document_inventory() -> list[dict[str, object]]:
    """Return actual files with active-collection chunk counts and job state."""
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        records = _metadata_records(collection())
    except Exception:
        records = []
    chunks_by_source: dict[str, int] = {}
    for _, metadata, _ in records:
        source = str(metadata.get("source", ""))
        if source:
            chunks_by_source[source] = chunks_by_source.get(source, 0) + 1
    manifest = _read_manifest()
    jobs = {str(job.get("filename")): job for job in indexing_snapshot()}
    inventory: list[dict[str, object]] = []
    for path in sorted(DOCUMENTS_DIR.glob("*.pdf"), key=lambda item: item.name.lower()):
        try:
            current_hash = file_hash(path)
            size = path.stat().st_size
        except OSError:
            continue
        job = jobs.get(path.name, {})
        chunks = chunks_by_source.get(path.name, 0)
        job_state = str(job.get("state", ""))
        if job_state in {"queued", "indexing"}:
            status = "Indexing" if job_state == "indexing" else "Queued"
        elif job_state == "failed" and str(job.get("file_hash", "")) == current_hash:
            status = "Failed"
        elif chunks > 0 and (manifest.get(path.name) == current_hash or not manifest.get(path.name)):
            status = "Indexed"
        else:
            status = "Pending"
        inventory.append(
            {
                "name": path.name,
                "size_bytes": size,
                "chunks": chunks,
                "status": status,
                "message": str(job.get("message", "")),
            }
        )
    return inventory


def document_metrics() -> dict[str, int]:
    """Calculate dashboard metrics from the active collection and disk files."""
    items = document_inventory()
    return {
        "total_documents": len(items),
        "total_chunks": sum(int(item["chunks"]) for item in items),
        "indexed_documents": sum(item["status"] == "Indexed" for item in items),
        "pending_or_failed": sum(item["status"] in {"Pending", "Queued", "Indexing", "Failed"} for item in items),
    }


def delete_document(filename: str) -> dict[str, int | str]:
    """Delete one managed PDF, its chunks in every provider collection, and manifests."""
    path = safe_document_path(filename)
    try:
        document_hash = file_hash(path) if path.exists() else ""
    except OSError as error:
        raise RuntimeError(f"Could not read {filename} before deletion.") from error

    with _STATE_LOCK:
        if document_hash:
            _CANCELLED_JOBS.add((filename, document_hash))

    removed_chunks = 0
    try:
        with _WRITE_LOCK:
            for target in _managed_collections():
                ids = _ids_for(target, {"source": filename})
                if ids:
                    target.delete(ids=ids)
                    removed_chunks += len(ids)
            for manifest_path in _all_manifest_paths():
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                except (FileNotFoundError, json.JSONDecodeError):
                    manifest = {}
                if filename in manifest:
                    manifest.pop(filename, None)
                    temporary = manifest_path.with_suffix(".tmp")
                    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
                    temporary.replace(manifest_path)
            if path.exists():
                path.unlink()
            elif removed_chunks == 0:
                raise FileNotFoundError(f"{filename} is no longer available.")
        _set_job(filename, state="deleted", message="Document and indexed chunks removed.", indexed_chunks=0)
        return {"filename": filename, "removed_chunks": removed_chunks}
    except Exception:
        with _STATE_LOCK:
            if document_hash:
                _CANCELLED_JOBS.discard((filename, document_hash))
        raise


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
        if _cancelled(filename, current_hash):
            _set_job(filename, state="deleted", message="Indexing cancelled because the document was deleted.", indexed_chunks=0)
            return
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
            # A file can be removed while an embedding request is in flight.
            # Never write those now-orphaned vectors back into Chroma.
            if _cancelled(filename, current_hash) or not path.exists():
                _set_job(filename, state="deleted", message="Indexing cancelled because the document was deleted.", indexed_chunks=0)
                return
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
            _CANCELLED_JOBS.discard((filename, current_hash))
            prior = _JOBS.get(filename, {})
            if prior.get("state") in {"queued", "indexing"} or (prior.get("state") == "failed" and prior.get("file_hash") == current_hash):
                continue
            _JOBS[filename] = {"filename": filename, "file_hash": current_hash, "state": "queued", "message": "Waiting to be indexed…", "indexed_chunks": 0}
        _INDEX_EXECUTOR.submit(_index_one_file, path)
        queued += 1
    return queued


def _quiz_context(selected_sources: list[str] | None, question_count: int) -> tuple[str, set[tuple[str, int]]]:
    """Build a bounded, source-labelled context from indexed chunks only."""
    records = _metadata_records(collection())
    if not records:
        raise RuntimeError("The knowledge base is empty. Upload and index a PDF before generating a quiz.")
    available_sources = {str(metadata.get("source", "")) for _, metadata, _ in records}
    requested = set(selected_sources or available_sources)
    if not requested or not requested.issubset(available_sources):
        raise RuntimeError("Choose documents that have finished indexing before generating a quiz.")

    grouped: dict[str, list[tuple[dict[str, object], str]]] = {}
    for _, metadata, text in records:
        source = str(metadata.get("source", ""))
        if source in requested and text:
            grouped.setdefault(source, []).append((metadata, text))
    for items in grouped.values():
        items.sort(key=lambda item: (int(item[0].get("page", 0)), int(item[0].get("chunk", 0))))

    # Round-robin selection keeps an all-documents quiz from being dominated by
    # the first large PDF while keeping the prompt bounded for both providers.
    # Keep the prompt small enough for the local llama3.2 context window while
    # still sampling enough distinct material for larger quizzes.
    limit = min(18, max(10, question_count))
    selected: list[tuple[dict[str, object], str]] = []
    positions = {source: 0 for source in sorted(grouped)}
    while len(selected) < limit:
        added = False
        for source in sorted(grouped):
            position = positions[source]
            if position < len(grouped[source]) and len(selected) < limit:
                selected.append(grouped[source][position])
                positions[source] += 1
                added = True
        if not added:
            break
    if not selected:
        raise RuntimeError("The selected documents do not have indexed text available for quiz generation.")

    valid_sources: set[tuple[str, int]] = set()
    blocks: list[str] = []
    for metadata, text in selected:
        source = str(metadata["source"])
        page = int(metadata["page"])
        valid_sources.add((source, page))
        blocks.append(f"[Source: {source}; Page: {page}]\n{text[:1400]}")
    return "\n\n".join(blocks), valid_sources


def _extract_json(text: str) -> object:
    """Extract JSON from a plain response or a Markdown code fence."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start_candidates = [position for position in (cleaned.find("["), cleaned.find("{")) if position >= 0]
    if start_candidates:
        cleaned = cleaned[min(start_candidates) :]
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as error:
        raise RuntimeError("The model returned an invalid quiz format. Please generate a new quiz.") from error


def _quiz_text(prompt: str) -> str:
    """Generate one quiz payload through the already selected provider."""
    cfg = config()
    client = None
    try:
        if cfg.provider == "ollama":
            response = ollama_client().chat.completions.create(
                model=cfg.chat_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You create accurate university study quizzes and return valid JSON only.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            return response.choices[0].message.content or ""

        from google.genai import types

        client = gemini_client()
        response = client.models.generate_content(
            model=cfg.chat_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction="You create accurate university study quizzes and return valid JSON only.",
                temperature=0.2,
                response_mime_type="application/json",
            ),
        )
        return response.text or ""
    except Exception as error:
        if cfg.provider == "ollama":
            raise RuntimeError("Ollama is not running or the configured model is unavailable. Start Ollama and try again.") from error
        message = _safe_gemini_error(error)
        LOGGER.warning("Gemini quiz request failed: %s", message)
        raise RuntimeError(f"Gemini quiz request failed: {message}") from error
    finally:
        if client is not None:
            try:
                client.close()
            except Exception as close_error:
                LOGGER.warning("Gemini client cleanup failed: %s", _safe_gemini_error(close_error))


def generate_quiz(selected_sources: list[str] | None, question_count: int, difficulty: str) -> list[dict[str, object]]:
    """Create validated, source-grounded MCQs from currently indexed chunks."""
    if question_count not in {5, 10, 15, 20}:
        raise ValueError("Choose 5, 10, 15, or 20 quiz questions.")
    if difficulty not in {"Easy", "Medium", "Hard"}:
        raise ValueError("Choose Easy, Medium, or Hard difficulty.")
    context, valid_sources = _quiz_context(selected_sources, question_count)
    prompt = f"""Create exactly {question_count} unique {difficulty}-difficulty multiple-choice questions using ONLY the university-document passages below.

Return a JSON array only. Each object must have exactly these fields:
question (string), options (object with keys A, B, C, D), correct_answer (one of A/B/C/D), explanation (string), source (exact source filename), page (integer).

Rules:
- Do not use general knowledge or facts not stated in the passages.
- Make each question distinct; do not repeat the same fact in different wording.
- All four options must be plausible, but only one may be correct.
- Keep the explanation short and grounded in the cited passage.
- Use only a source filename and page pair that appears in the context labels.

Document passages:
{context}"""
    payload = _extract_json(_quiz_text(prompt))
    entries = payload.get("questions", []) if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        raise RuntimeError("The model returned an invalid quiz format. Please generate a new quiz.")

    questions: list[dict[str, object]] = []
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        question = str(entry.get("question", "")).strip()
        options = entry.get("options")
        correct_answer = str(entry.get("correct_answer", "")).strip().upper()
        explanation = str(entry.get("explanation", "")).strip()
        source = str(entry.get("source", "")).strip()
        try:
            page = int(entry.get("page"))
        except (TypeError, ValueError):
            continue
        normalised = re.sub(r"\W+", "", question.lower())
        if (
            not question
            or not isinstance(options, dict)
            or set(options) != {"A", "B", "C", "D"}
            or any(not str(options[key]).strip() for key in ("A", "B", "C", "D"))
            or correct_answer not in {"A", "B", "C", "D"}
            or not explanation
            or (source, page) not in valid_sources
            or normalised in seen
        ):
            continue
        seen.add(normalised)
        questions.append(
            {
                "question": question,
                "options": {key: str(options[key]).strip() for key in ("A", "B", "C", "D")},
                "correct_answer": correct_answer,
                "explanation": explanation,
                "source": source,
                "page": page,
            }
        )
    if len(questions) != question_count:
        raise RuntimeError("The model did not return enough valid, source-grounded questions. Please generate a new quiz.")
    return questions


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
