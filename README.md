# RAG-Based University Academic Assistant

A Streamlit chatbot that searches university PDFs before using an LLM to answer. It supports local Ollama development and Gemini on Streamlit Cloud, with source/page citations for every answer.

## What it does

1. Accepts university PDF documents in the interface (or in `data/documents/`).
2. Extracts page text and splits it into overlapping chunks.
3. Creates embeddings with the active provider and saves them in ChromaDB.
4. Retrieves the most relevant passages for each question.
5. Generates a grounded answer with the active provider and document/page citations.
6. Indexes new or changed PDFs in one background worker, so existing indexed documents remain searchable during indexing.

## Setup

Use Python 3.10 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Upload PDFs in the sidebar. New or modified files are queued for background indexing automatically; the **Check and index new PDFs** button performs the same non-blocking check manually. The app never deletes the ChromaDB collection during normal use.

## Local Ollama setup

The included `.env` is configured for Ollama. Run an Ollama server and ensure these models are available:

```powershell
ollama pull llama3.2
ollama pull nomic-embed-text
```

Do not change `LLM_PROVIDER`, `OLLAMA_CHAT_MODEL`, or `OLLAMA_EMBEDDING_MODEL` unless you deliberately choose different local Ollama models.

Start the application:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Streamlit Cloud setup

1. Push this repository to GitHub and create a Streamlit Cloud app pointing to `app.py`.
2. In the app dashboard, open **App settings → Secrets** and add the following TOML. Replace only the placeholder with your real key; never commit that key.

```toml
LLM_PROVIDER = "gemini"
GEMINI_API_KEY = "your-gemini-api-key"
GEMINI_CHAT_MODEL = "gemini-3.5-flash-lite"
GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
TOP_K = "3"
CHUNK_SIZE = "1200"
CHUNK_OVERLAP = "100"
```

3. Save the secrets and redeploy. Streamlit Cloud will use Gemini and will never attempt to connect to your local Ollama instance.

The Gemini deployment uses a separate Chroma collection and manifest from the local Ollama index because their embedding dimensions differ.

## Important safeguards

- Use only approved and current university documents.
- Changed PDFs are detected by SHA-256 hash and re-indexed without rebuilding unrelated documents.
- Scanned/image-only PDFs need OCR before indexing.
- The tool is informational; important decisions should be verified with the responsible university office.
- Gemini free-tier quotas and model availability can limit large embedding jobs. If a request is rejected, wait for quota reset or use a billing-enabled Gemini project.

## Example questions

- What is the minimum attendance requirement?
- How many credits are required to graduate?
- What is the procedure for applying for revaluation?
- When is the last date for course registration?
