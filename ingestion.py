import subprocess
from pathlib import Path

import chromadb
import tree_sitter_cpp
from sentence_transformers import SentenceTransformer
from tree_sitter import Language, Parser, Query, QueryCursor


query_logic = """
(preproc_include) @include
(class_specifier) @class
(struct_specifier) @struct
(function_definition) @function
(declaration) @global_or_prototype
"""


def clone_repo(repo_url: str, repo_folder: str | Path = "repo") -> Path:
	repo_path = Path(repo_folder)

	if repo_path.exists() and any(repo_path.iterdir()):
		raise FileExistsError(f"repo folder already exists and is not empty: {repo_path}")

	repo_path.mkdir(parents=True, exist_ok=True)
	command = ["git", "clone", repo_url, str(repo_path)]
	result = subprocess.run(command, capture_output=True, text=True, check=False)

	if result.returncode != 0:
		message = result.stderr.strip() or result.stdout.strip() or "git command failed"
		raise RuntimeError(message)

	return repo_path


def ingest_repo(repo_url: str, repo_folder: str | Path = "repo", chroma_folder: str | Path = "chroma_db") -> str:
	repo_path = clone_repo(repo_url, repo_folder)

	cpp_files = [
		path for path in repo_path.rglob("*")
		if path.is_file() and path.suffix.lower() in {".cpp", ".cc", ".cxx", ".hpp", ".hh", ".h", ".hxx", ".ipp"}
	]
	if not cpp_files:
		raise FileNotFoundError("no C++ files found in cloned repository")

	language = Language(tree_sitter_cpp.language())
	parser = Parser()
	parser.language = language
	query = Query(language, query_logic)
	embeddings_model = SentenceTransformer(
		"jinaai/jina-embeddings-v2-base-code",
		model_kwargs={"attn_implementation": "eager"},
	)
	client = chromadb.PersistentClient(path=str(chroma_folder))
	collection = client.get_or_create_collection(name="cpp_chunks")

	documents = []
	metadatas = []
	ids = []

	for file_path in cpp_files:
		source = file_path.read_bytes()
		tree = parser.parse(source)
		cursor = QueryCursor(query)
		chunk_index = 0
		for _, captures in cursor.matches(tree.root_node):
			for capture_name, nodes in captures.items():
				for node in nodes:
					chunk = source[node.start_byte:node.end_byte].decode("utf-8", errors="ignore").strip()
					if not chunk:
						continue
					documents.append(chunk)
					start_line = node.start_point[0] + 1
					end_line = node.end_point[0] + 1
					metadatas.append({
						"file": str(file_path),
						"capture": capture_name,
						"chunk_index": chunk_index,
						"start_line": start_line,
						"end_line": end_line,
					})
					ids.append(f"{file_path.as_posix()}::{chunk_index}")
					chunk_index += 1

	if not documents:
		print("no chunks found")
		return

	embeddings = embeddings_model.encode(documents, normalize_embeddings=True).tolist()
	collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
	return f"cloned {repo_url} into {repo_path} and stored {len(documents)} chunks in chroma at {chroma_folder}"


if __name__ == "__main__":
	raise SystemExit("Use ingest_repo(repo_url) from another module or a small runner script.")
