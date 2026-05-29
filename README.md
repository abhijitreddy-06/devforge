# DevForge

DevForge is a Python workspace for cloning a GitHub repository, ingesting C++ code into Chroma with Tree-sitter and Jina embeddings, and then running a LangGraph-based read-only RAG loop over the stored code.

## What This Project Does

1. Clones a GitHub repo into a local `repo/` folder.
2. Parses C++ files with Tree-sitter and stores code chunks in Chroma.
3. Searches the indexed codebase with the same embedding model used for ingestion.
4. Runs a LangGraph agent that searches the codebase, reads targeted file lines, and answers from the retrieved evidence.
5. Uses a simple Streamlit UI to submit a codebase question, with secrets loaded from `.env`.

## Main Files

- `app.py` - Streamlit entrypoint for the read-only RAG UI.
- `orchestrator.py` - LangGraph agent that coordinates search and file reads.
- `reterieval_tools.py` - tool collection for search and file reading.
- `ingestion.py` - clone + Tree-sitter + Chroma ingestion pipeline for C++ files.
- `repo/` - local cloned source tree used during ingestion.
- `chroma_db/` - local persistent Chroma database.

## Tools In The Project

- `ingest_repo(repo_url, repo_folder, chroma_folder)` - clones a repo into `repo/`, chunks and embeds C++ files into Chroma.
- `search_codebase(query)` - searches Chroma for relevant code chunks.
- `read_file_lines(file_path, start_line, end_line)` - reads only a requested line range.

## Workflow

### Stage 1 - Ingest

The app asks for a GitHub URL, clones the repo into `repo/`, and ingests C++ files into Chroma.

### Stage 2 - Ask

After ingestion succeeds, the app asks for a question about the codebase.

The Mistral API key is loaded from `.env` instead of being typed into the frontend.

The LangGraph agent then loops through:

1. search the codebase
2. inspect the relevant file lines
3. search or read again if the evidence is not enough
4. answer the user once it has enough context

## Environment

- Python 3.11
- Poetry
- Mistral API key in `.env`

## Setup

Install dependencies and run the app:

```powershell
poetry install
poetry run streamlit run app.py
```

Create your local `.env` from the example file first:

```powershell
Copy-Item .env.example .env
```

Then fill in `MISTRAL_API_KEY` and optionally `MISTRAL_MODEL`.

## Notes

- `repo/` and `chroma_db/` are runtime folders and are ignored by Git.
- The embedding model is `jinaai/jina-embeddings-v2-base-code`.
- The agent uses Mistral instead of Gemini.
- `.env.example` contains the supported environment variables for local setup.

## Current State

The project already contains working implementations for cloning, ingestion, and retrieval. The current Streamlit flow is now a single read-only RAG screen that answers questions from the indexed codebase.
