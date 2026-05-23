import os

from langchain_litellm import ChatLiteLLM


DEFAULT_MODEL = "anthropic/claude-sonnet-4-5"


def _required_api_key(model_id: str) -> str | None:
    provider = model_id.split("/", 1)[0].lower()
    return {
        "anthropic": "ANTHROPIC_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "openai": "OPENAI_API_KEY",
    }.get(provider)


def build_llm(model: str | None = None, **kwargs) -> ChatLiteLLM:
    """Return a ChatLiteLLM bound to the configured model.

    Model id follows LiteLLM conventions: ``provider/model``. Override per call or set
    ``TRAIL_BUDDY_MODEL`` in the environment. Provider API keys are read by LiteLLM
    from the usual env vars (``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY``, etc.).
    """
    model_id = model or os.getenv("TRAIL_BUDDY_MODEL", DEFAULT_MODEL)
    api_key_var = _required_api_key(model_id)
    if api_key_var and not os.getenv(api_key_var):
        model_source = "TRAIL_BUDDY_MODEL" if model is None else "model"
        raise RuntimeError(
            f"{api_key_var} is required for {model_source}={model_id!r}. "
            "Set it in .env or choose a model/provider that does not need this key."
        )
    temperature = kwargs.pop("temperature", 0.4)
    return ChatLiteLLM(model=model_id, temperature=temperature, **kwargs)
