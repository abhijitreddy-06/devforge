import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage

from ingestion import ingest_repo
from orchestrator import run_agent


load_dotenv()


def _extract_final_response(messages):
	for message in reversed(messages or []):
		if isinstance(message, AIMessage):
			content = message.content
			return content if isinstance(content, str) else str(content)
	return "No final response was returned."


def _render_result(result):
	if result.get("error"):
		st.error(result["error"])
		return

	messages = result.get("messages", [])
	show_reasoning = st.toggle("Show complete reasoning", value=False)

	if show_reasoning:
		st.subheader("Complete reasoning")
		for message in messages:
			message_type = message.__class__.__name__.replace("Message", "")
			st.markdown(f"**{message_type}**")
			st.write(message.content)
	else:
		st.subheader("Final response")
		st.write(_extract_final_response(messages))


def main():
	st.set_page_config(page_title="Devforge RAG", page_icon="🔎", layout="centered")
	st.title("Devforge RAG")
	st.write("Stage the repository once, then ask questions against the indexed codebase. The agent only searches and reads files until it has enough evidence to answer.")

	if "stage" not in st.session_state:
		st.session_state.stage = "ingest"

	if "last_result" not in st.session_state:
		st.session_state.last_result = None

	if st.session_state.stage == "ingest":
		st.subheader("1. Ingest repository")
		repo_url = st.text_input("GitHub repo URL")

		col1, col2 = st.columns(2)
		with col1:
			clone_clicked = st.button("Clone and Ingest")
		with col2:
			already_ingested_clicked = st.button("I already have ingested")

		if clone_clicked:
			if not repo_url.strip():
				st.error("Please enter a GitHub repo URL.")
			else:
				with st.spinner("Cloning and ingesting the repository..."):
					try:
						ingest_repo(repo_url.strip(), repo_folder="repo", chroma_folder="chroma_db")
						st.session_state.stage = "ask"
						st.session_state.last_result = None
						st.rerun()
					except Exception as exc:
						st.error(f"Error: {exc}")

		if already_ingested_clicked:
			st.session_state.stage = "ask"
			st.session_state.last_result = None
			st.rerun()

	elif st.session_state.stage == "ask":
		st.subheader("2. Ask a codebase question")
		st.write("The agent searches the indexed codebase, reads the exact file lines it needs, and answers only from retrieved evidence.")

		query = st.text_area("Question", value="Where is the division logic located?", height=120)

		if st.button("Search codebase"):
			if not query.strip():
				st.error("Please enter a question.")
			else:
				with st.spinner("Searching the codebase..."):
					try:
						st.session_state.last_result = run_agent(query.strip())
					except Exception as exc:
						st.session_state.last_result = {"error": str(exc)}

		if st.session_state.last_result:
			_render_result(st.session_state.last_result)


if __name__ == "__main__":
	main()
