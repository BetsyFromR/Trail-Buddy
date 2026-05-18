from __future__ import annotations

import logging
import os
import uuid

import gradio as gr
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

load_dotenv()

from trail_buddy.graph import build_graph
from trail_buddy.logging_config import configure_logging

configure_logging()

logger = logging.getLogger(__name__)

graph = build_graph()


def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts)
    return str(content or "")


def chat_fn(message: str, history: list, session_id: str):
    config = {"configurable": {"thread_id": session_id}}
    inputs = {"messages": [HumanMessage(content=message)]}

    accumulated = ""
    try:
        for chunk, _meta in graph.stream(inputs, config, stream_mode="messages"):
            # Skip tool messages and tool-call requests; only advisor text
            # should reach the chat bubble.
            if isinstance(chunk, AIMessage) and getattr(chunk, "tool_calls", None):
                continue
            if not isinstance(chunk, (AIMessage, AIMessageChunk)):
                continue
            token = _content_text(getattr(chunk, "content", ""))
            if token:
                if isinstance(chunk, AIMessage):
                    accumulated = token
                else:
                    accumulated += token
                yield accumulated
    except Exception:
        logger.exception("Chat response streaming failed")
        raise


def new_session() -> str:
    return uuid.uuid4().hex


with gr.Blocks(title="Trail Buddy") as demo:
    gr.Markdown("# 🏔 Trail Buddy\nAsk about trail races, gear, training, recovery.")
    session_id = gr.State(value=new_session)
    gr.ChatInterface(
        fn=chat_fn,
        additional_inputs=[session_id],
        examples=[
            ["I want to run a trail. Before I only run half marathons. Do you think I can cope with 35 km with 2000 elevation in Montenegro trail?"],
            ["How to choose poles for trailrunning?"],
            ["Should I wash the sneakers?"],
        ],
    )


if __name__ == "__main__":
    demo.launch(server_name=os.getenv("GRADIO_HOST", "127.0.0.1"), quiet=True)
