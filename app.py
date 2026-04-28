from __future__ import annotations

import os
import uuid

import gradio as gr
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from trail_buddy.graph import build_graph

load_dotenv()

graph = build_graph()


def chat_fn(message: str, history: list, session_id: str):
    config = {"configurable": {"thread_id": session_id}}
    inputs = {"messages": [HumanMessage(content=message)]}

    accumulated = ""
    for chunk, _meta in graph.stream(inputs, config, stream_mode="messages"):
        token = getattr(chunk, "content", "") or ""
        if token:
            accumulated += token
            yield accumulated


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
            ["А кроссовки надо стирать?"],
        ],
    )


if __name__ == "__main__":
    demo.launch(server_name=os.getenv("GRADIO_HOST", "127.0.0.1"))
