"""Drive a couple of representative queries through the graph so we can read
the new tool/error/citation contract in the logs. Throwaway script."""
from __future__ import annotations

import json
import logging
import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from trail_buddy.graph import build_graph
from trail_buddy.logging_config import configure_logging

load_dotenv()


def run(label: str, prompt: str, thread: str) -> None:
    logging.getLogger(__name__).info("[smoke] >>> %s :: %r", label, prompt)
    graph = build_graph()
    out = graph.invoke(
        {"messages": [HumanMessage(content=prompt)]},
        config={"configurable": {"thread_id": thread}},
    )
    msgs = out["messages"]
    for m in msgs:
        kind = getattr(m, "type", type(m).__name__)
        content = getattr(m, "content", "")
        artifact = getattr(m, "artifact", None)
        line = f"[smoke] {label} :: {kind}: {str(content)[:600]}"
        logging.getLogger(__name__).info(line)
        if artifact is not None:
            logging.getLogger(__name__).info(
                "[smoke] %s :: artifact=%s", label, json.dumps(artifact)[:600]
            )
    logging.getLogger(__name__).info("[smoke] <<< %s done", label)


def main() -> None:
    configure_logging()
    os.environ.setdefault("TRAIL_BUDDY_LOG_LEVEL", "INFO")
    run(
        label="ambiguous",
        prompt="What's the weather in Boka Bay this weekend? I want to run a trail there.",
        thread="smoke-amb",
    )
    run(
        label="citation",
        prompt="How should I choose trekking poles for steep technical descents?",
        thread="smoke-cite",
    )


if __name__ == "__main__":
    main()
