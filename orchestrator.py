import os
import time
from functools import lru_cache
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_mistralai import ChatMistralAI
from langgraph.graph import START, StateGraph, add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from reterieval_tools import read_file_lines, search_codebase

load_dotenv()


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    turn_count: int

TOOLS = [search_codebase, read_file_lines]

SYSTEM_PROMPT = (
    "You are a codebase RAG assistant for a C++ repository. "
    "Use search_codebase to retrieve relevant chunks, then judge whether the returned chunks already answer the question. "
    "If a retrieved chunk is sufficient, stop using tools immediately and answer from that evidence. "
    "Do not call read_file_lines just because it is available. "
    "Only call read_file_lines when you need exact surrounding lines, more context, or disambiguation. "
    "Do not repeatedly search for the same information. "
    "If sufficient evidence exists, stop using tools and answer. "
    "If any tool returns TOOL_GUARD_BLOCKED or TOOL_LOOP_DETECTED, stop calling tools immediately and provide the best possible answer from existing evidence. "
    "Do not write files, compile code, or mention unsupported actions. "
    "If the retrieved evidence is enough, answer directly and briefly. "
    "If it is not enough, gather more context with another tool call. "
    "Keep the response grounded in the retrieved code and cite file paths or line ranges when possible."
)


@lru_cache(maxsize=1)
def _get_llm() -> ChatMistralAI:
    return ChatMistralAI(
        model=os.getenv("MISTRAL_MODEL", "mistral-small-latest"),
        temperature=0,
    )


@lru_cache(maxsize=1)
def _build_app():
    llm_with_tools = _get_llm().bind_tools(TOOLS)

    def agent_node(state: AgentState):
        if state.get("turn_count", 0) > 0:
            delay_seconds = float(os.getenv("DEVFORGE_LLM_DELAY_SECONDS", "0.5"))
            if delay_seconds > 0:
                time.sleep(delay_seconds)

        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response], "turn_count": state.get("turn_count", 0) + 1}

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools=TOOLS))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")

    return graph.compile()


def run_agent(query: str):
    """Run the read-only codebase RAG loop."""
    app = _build_app()
    max_turns = int(os.getenv("DEVFORGE_MAX_TOOL_TURNS", 20))

    try:
        return app.invoke(
            {"messages": [HumanMessage(content=query)], "turn_count": 0},
            config={"recursion_limit": max_turns * 2},
        )
    except Exception as exc:
        if "Recursion limit" in str(exc):
            return {"error": f"Agent stopped after {max_turns} turns without a final answer."}
        raise


if __name__ == "__main__":
    response = run_agent("Where is the division logic located?")
    final_message = response["messages"][-1].content if response.get("messages") else response
    print(final_message)