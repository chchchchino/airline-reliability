"""Gradio chat UI for the airline delay LangGraph agent."""

from __future__ import annotations

import asyncio
import os
from typing import Any

import gradio as gr
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

from airline_reliability.graph import build_graph

load_dotenv()

_graph = None
_graph_lock = asyncio.Lock()


async def _get_graph():
    global _graph
    async with _graph_lock:
        if _graph is None:
            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY is not set. Add it to .env or the environment.")
            _graph = await build_graph()
        return _graph


def _text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content) if content is not None else ""


def _history_to_messages(history: list[Any], user_message: str) -> list:
    """Gradio 6 ChatInterface passes history as OpenAI-style dicts with role/content."""
    messages: list = []
    for turn in history or []:
        if isinstance(turn, dict):
            role = turn.get("role")
            text = _text_from_content(turn.get("content")).strip()
            if not text:
                continue
            if role == "user":
                messages.append(HumanMessage(content=text))
            elif role == "assistant":
                messages.append(AIMessage(content=text))
        elif isinstance(turn, (list, tuple)):
            user_part, bot_part = (turn + ("", ""))[:2]
            if user_part is not None and str(user_part).strip():
                messages.append(HumanMessage(content=str(user_part)))
            if bot_part is not None and str(bot_part).strip():
                messages.append(AIMessage(content=str(bot_part)))
    messages.append(HumanMessage(content=user_message))
    return messages


def _last_ai_text(messages: list) -> str:
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            c = m.content
            if isinstance(c, str) and c.strip():
                return c
            if isinstance(c, list):
                parts = []
                for block in c:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(block.get("text", ""))
                text = "".join(parts).strip()
                if text:
                    return text
    return "(No text response.)"


async def chat_fn(message: str, history: list):
    graph = await _get_graph()
    lc_messages = _history_to_messages(history, message)
    result = await graph.ainvoke({"messages": lc_messages})
    return _last_ai_text(result["messages"])


def main() -> None:
    demo = gr.ChatInterface(
        fn=chat_fn,
        title="Airline delay assistant",
        description="Ask questions about U.S. airline delays. The agent uses MCP tools with RAG over the bundled CSV.",
    )
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")))


if __name__ == "__main__":
    main()
