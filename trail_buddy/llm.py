import os

from langchain_litellm import ChatLiteLLM


DEFAULT_MODEL = "anthropic/claude-sonnet-4-5"


def build_llm(model: str | None = None, **kwargs) -> ChatLiteLLM:
    """Return a ChatLiteLLM bound to the configured model.

    Model id follows LiteLLM conventions: ``provider/model``. Override per call or set
    ``TRAIL_BUDDY_MODEL`` in the environment. Provider API keys are read by LiteLLM
    from the usual env vars (``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY``, etc.).
    """
    model_id = model or os.getenv("TRAIL_BUDDY_MODEL", DEFAULT_MODEL)
    temperature = kwargs.pop("temperature", 0.4)
    return ChatLiteLLM(model=model_id, temperature=temperature, **kwargs)
