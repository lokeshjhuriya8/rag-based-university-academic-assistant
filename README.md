# RAG-Based University Academic Assistant

A local Streamlit chatbot that searches university PDFs before using an LLM to answer. It returns source and page citations and is instructed to decline questions that are not supported by the indexed documents.

## What it does

1. Accepts university PDF documents in the interface (or in `data/documents/`).
2. Extracts page text and splits it into overlapping chunks.
3. Creates embeddings with local Ollama (`nomic-embed-text`) and saves them in ChromaDB.
4. Retrieves the most relevant passages for each question.
5. Generates a grounded answer with Llama 3.2 and document/page citations.
6. Indexes new or changed PDFs in one background worker, so existing indexed documents remain searchable during indexing.

## Setup

Use Python 3.10 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Keep the existing Ollama settings in `.env`, start Ollama, then run:

```powershell
streamlit run app.py
```

Upload PDFs in the sidebar. New or modified files are queued for background indexing automatically; the **Check and index new PDFs** button performs the same non-blocking check manually. The app never deletes the ChromaDB collection during normal use.

## Ollama setup

The included `.env` is configured for Ollama. Run an Ollama server and ensure these models are available:

```powershell
ollama pull llama3.2
ollama pull nomic-embed-text
```

Do not change `LLM_PROVIDER`, `OLLAMA_CHAT_MODEL`, or `OLLAMA_EMBEDDING_MODEL` unless you deliberately choose different local Ollama models.

## Important safeguards

- Use only approved and current university documents.
- Changed PDFs are detected by SHA-256 hash and re-indexed without rebuilding unrelated documents.
- Scanned/image-only PDFs need OCR before indexing.
- The tool is informational; important decisions should be verified with the responsible university office.

## Example questions

- What is the minimum attendance requirement?
- How many credits are required to graduate?
- What is the procedure for applying for revaluation?
- When is the last date for course registration?
