"""LangGraph agent with MCP-backed tools (stdio airline delay server)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from mcp.client.stdio import get_default_environment


SYSTEM_PROMPT = """You are an analyst for U.S. airline on-time and delay reporting data.
The data lives in a large CSV; tools use RAG (semantic search) — you only receive relevant chunks,
not the full file. Use get_airline_delay_dataset_schema if you need column names, then
search_airline_delay_causes with focused queries (carrier, airport code, delay type, etc.).
You may call search multiple times. Answer from retrieved text; say if evidence is missing."""


def _mcp_subprocess_env() -> dict[str, str]:
    """MCP stdio only inherits a small allowlisted env unless we pass env explicitly."""
    env = dict(get_default_environment())
    for key in (
        "OPENAI_API_KEY",
        "AIRLINE_DELAY_CSV_PATH",
        "AIRLINE_DELAY_EMBED_MODEL",
        "AIRLINE_DELAY_RAG_ROWS_PER_CHUNK",
    ):
        val = os.environ.get(key)
        if val is not None:
            env[key] = val
    return env


async def build_graph():
    """Compile a LangGraph that uses the local MCP stdio server for CSV access."""
    project_root = Path(__file__).resolve().parents[2]
    server_module = "airline_reliability.mcp_server"

    client = MultiServerMCPClient(
        {
            "airline_delays": {
                "command": sys.executable,
                "args": ["-m", server_module],
                "transport": "stdio",
                "cwd": str(project_root),
                "env": _mcp_subprocess_env(),
            },
        }
    )
    tools = await client.get_tools()

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    model_with_tools = model.bind_tools(tools)

    async def call_model(state: MessagesState) -> dict[str, list[BaseMessage]]:
        messages = list(state["messages"])
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT), *messages]
        response = await model_with_tools.ainvoke(messages)
        return {"messages": [response]}

    builder = StateGraph(MessagesState)
    builder.add_node("call_model", call_model)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges("call_model", tools_condition)
    builder.add_edge("tools", "call_model")

    return builder.compile()
