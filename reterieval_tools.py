from pathlib import Path
from collections import deque
from functools import lru_cache
from threading import Lock
import re

from langchain_core.tools import tool

CHROMA_PATH = Path("chroma_db")
COLLECTION_NAME = "cpp_chunks"
MODEL_NAME = "jinaai/jina-embeddings-v2-base-code"

RECENT_SEARCH_WINDOW = 16
RECENT_READ_WINDOW = 24
CALL_HISTORY_WINDOW = 12
CPP_SUFFIXES = {".cpp", ".cc", ".cxx", ".hpp", ".hh", ".h", ".hxx", ".ipp"}
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "code", "does", "for", "from",
    "how", "in", "is", "it", "located", "logic", "of", "on", "or", "the", "to", "what",
    "where", "which", "with",
}

_guard_lock = Lock()
_recent_search_queries = deque(maxlen=RECENT_SEARCH_WINDOW)
_recent_read_ranges = deque(maxlen=RECENT_READ_WINDOW)
_call_history = deque(maxlen=CALL_HISTORY_WINDOW)
_loop_termination_active = False


@lru_cache(maxsize=1)
def _get_collection():
    """Lazy-loads the embedding model and DB connection once per session."""
    import chromadb
    import torch
    from sentence_transformers import SentenceTransformer

    embedding_model = SentenceTransformer(
        MODEL_NAME,
        device="cpu",
        model_kwargs={"attn_implementation": "eager", "torch_dtype": torch.float32},
    )
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    return embedding_model, collection


def _relevance_label(distance: float | None) -> str:
    if distance is None:
        return "unknown"
    if distance <= 0.25:
        return "high"
    if distance <= 0.5:
        return "medium"
    return "low"


def _relevance_label_from_score(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.45:
        return "medium"
    return "low"


def _normalize_query(query: str) -> str:
    return " ".join(query.strip().lower().split())


def _normalize_path(file_path: str) -> str:
    return str(Path(file_path)).replace("\\", "/").lower()


def _extract_terms(query: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", query.lower())
    return [token for token in tokens if token not in STOP_WORDS and len(token) > 1]


def _expanded_queries(query: str) -> list[str]:
    base = _normalize_query(query)
    variants = [base]

    terms = _extract_terms(base)
    if terms:
        variants.append(" ".join(terms[:8]))
        variants.append(f"implementation location {' '.join(terms[:6])}")

    # Keep order, drop duplicates.
    deduped: list[str] = []
    for item in variants:
        if item and item not in deduped:
            deduped.append(item)
    return deduped


def _score_lexical_line(line: str, query_terms: list[str]) -> float:
    if not query_terms:
        return 0.0

    lowered = line.lower()
    token_hits = sum(1 for term in query_terms if term in lowered)
    token_score = token_hits / max(1, len(query_terms))
    return min(1.0, token_score)


def _lexical_fallback(query: str, max_results: int = 5) -> list[dict]:
    repo_root = Path("repo")
    if not repo_root.exists():
        return []

    terms = _extract_terms(query)
    if not terms:
        return []

    candidates: list[dict] = []
    for file_path in repo_root.rglob("*"):
        if not file_path.is_file() or file_path.suffix.lower() not in CPP_SUFFIXES:
            continue
        lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        for idx, line in enumerate(lines, start=1):
            score = _score_lexical_line(line, terms)
            if score <= 0:
                continue
            start_line = max(1, idx - 1)
            end_line = min(len(lines), idx + 1)
            chunk = "\n".join(lines[start_line - 1:end_line])
            candidates.append(
                {
                    "id": f"lexical::{file_path.as_posix()}::{idx}",
                    "distance": None,
                    "similarity": score,
                    "source": "lexical",
                    "metadata": {
                        "file": str(file_path),
                        "start_line": start_line,
                        "end_line": end_line,
                    },
                    "chunk": chunk,
                }
            )

    candidates.sort(key=lambda item: item.get("similarity", 0.0), reverse=True)
    return candidates[:max_results]


def _semantic_results(query: str, n_results: int = 10) -> list[dict]:
    model, collection = _get_collection()
    merged_items: list[dict] = []

    for variant in _expanded_queries(query):
        query_embedding = model.encode([variant], normalize_embeddings=True)[0].tolist()
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        ids = result.get("ids", [[]])[0]
        if not ids:
            continue

        for cid, doc, meta, dist in zip(
            ids,
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            similarity = max(0.0, 1.0 - float(dist)) if dist is not None else 0.0
            merged_items.append(
                {
                    "id": cid,
                    "distance": dist,
                    "similarity": similarity,
                    "source": "semantic",
                    "metadata": meta,
                    "chunk": doc,
                }
            )

    return merged_items


def _dedupe_and_rank(items: list[dict], top_k: int = 5) -> list[dict]:
    by_key: dict[str, dict] = {}

    for item in items:
        metadata = item.get("metadata") or {}
        file_path = _normalize_path(metadata.get("file") or metadata.get("file_path") or "unknown")
        start_line = metadata.get("start_line")
        end_line = metadata.get("end_line")
        chunk = (item.get("chunk") or "").strip()
        key = f"{file_path}:{start_line}:{end_line}:{chunk[:120]}"

        current = by_key.get(key)
        if current is None or item.get("similarity", 0.0) > current.get("similarity", 0.0):
            by_key[key] = item

    ranked = sorted(by_key.values(), key=lambda x: x.get("similarity", 0.0), reverse=True)
    return ranked[:top_k]


def _guard_message(reason: str) -> str:
    return (
        "TOOL_GUARD_BLOCKED: "
        f"{reason} "
        "Stop repeating tool calls. Use existing evidence to answer, or explain what is missing."
    )


def _loop_message() -> str:
    return (
        "TOOL_LOOP_DETECTED: repeated tool-call pattern detected. "
        "Terminate tool usage now and provide the best possible answer from gathered evidence."
    )


def _has_repeated_pattern(history: list[str]) -> bool:
    if len(history) >= 3 and history[-1] == history[-2] == history[-3]:
        return True
    if len(history) >= 4 and history[-4:-2] == history[-2:]:
        return True
    if len(history) >= 6 and history[-6:-3] == history[-3:]:
        return True
    return False


def _register_tool_call(tool_name: str, key: str) -> str | None:
    global _loop_termination_active

    with _guard_lock:
        if _loop_termination_active:
            return _loop_message()

        call_signature = f"{tool_name}:{key}"

        if tool_name == "search" and key in _recent_search_queries:
            _call_history.append(call_signature)
            return _guard_message("Duplicate search query blocked.")

        if tool_name == "read" and key in _recent_read_ranges:
            _call_history.append(call_signature)
            return _guard_message("Duplicate file/range read blocked.")

        if tool_name == "search":
            _recent_search_queries.append(key)
        elif tool_name == "read":
            _recent_read_ranges.append(key)

        _call_history.append(call_signature)
        if _has_repeated_pattern(list(_call_history)):
            _loop_termination_active = True
            return _loop_message()

    return None


def _format_search_result(index: int, item: dict) -> str:
    metadata = item.get("metadata") or {}
    file_path = metadata.get("file") or metadata.get("file_path") or "unknown"
    start_line = metadata.get("start_line")
    end_line = metadata.get("end_line")
    distance = item.get("distance")
    similarity = item.get("similarity")
    source = item.get("source", "semantic")
    snippet = (item.get("chunk") or "").strip()
    snippet_preview = snippet if len(snippet) <= 700 else f"{snippet[:700]}..."

    follow_up = "unknown"
    if start_line is not None and end_line is not None:
        follow_up = (
            f'read_file_lines(file_path="{file_path}", start_line={start_line}, end_line={end_line})'
        )

    return (
        f"Result {index}\n"
        f"  source: {source}\n"
        f"  file_path: {file_path}\n"
        f"  chunk_start_line: {start_line if start_line is not None else 'unknown'}\n"
        f"  chunk_end_line: {end_line if end_line is not None else 'unknown'}\n"
        f"  start_line: {start_line if start_line is not None else 'unknown'}\n"
        f"  end_line: {end_line if end_line is not None else 'unknown'}\n"
        f"  distance: {distance if distance is not None else 'unknown'}\n"
        f"  similarity: {round(similarity, 4) if isinstance(similarity, (int, float)) else 'unknown'}\n"
        f"  relevance: {_relevance_label(distance) if source == 'semantic' else _relevance_label_from_score(float(similarity or 0.0))}\n"
        f"  follow_up_read: {follow_up}\n"
        f"  chunk:\n{snippet_preview}"
    )


@tool
def search_codebase(query: str) -> str:
    """Search the indexed C++ chunks and return human-readable ranked results.

    Use this tool first for every question. The output includes `file_path`, `start_line`,
    `end_line`, a `relevance` label, and the exact `read_file_lines(...)` call to use only
    if more context is needed. If a result already answers the question, stop here and answer.
    """
    if not CHROMA_PATH.exists():
        return "Error: chroma_db folder does not exist."

    query_key = _normalize_query(query)
    guard_result = _register_tool_call("search", query_key)
    if guard_result is not None:
        return guard_result

    semantic_items = _semantic_results(query, n_results=8)
    lexical_items = _lexical_fallback(query, max_results=8)
    ranked_items = _dedupe_and_rank(semantic_items + lexical_items, top_k=5)

    if not ranked_items:
        return "No chunks found."

    rendered_items = [
        _format_search_result(index + 1, item)
        for index, item in enumerate(ranked_items)
    ]
    return "\n\n".join(rendered_items)


@tool
def read_file_lines(file_path: str, start_line: int, end_line: int) -> str:
    """Read exactly one inclusive line range from a file.

    Use this only after search_codebase if the returned chunk is not enough. Pass the exact
    `file_path`, `start_line`, and `end_line` from the search result. If the search result
    already answers the question, do not call this tool.
    """
    read_key = f"{_normalize_path(file_path)}:{start_line}:{end_line}"
    guard_result = _register_tool_call("read", read_key)
    if guard_result is not None:
        return guard_result

    path = Path(file_path)
    if not path.exists():
        return "Error: file does not exist."
    if start_line < 1 or end_line < start_line:
        return "Error: invalid line range."

    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return "\n".join(lines[start_line - 1 : end_line])
