from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "evaluation" / "eval_config.toml"


@dataclass(frozen=True)
class EvaluationSettings:
    dataset_path: Path
    output_dir: Path
    agent_model: str = "anthropic/claude-sonnet-4-5"
    judge_model: str = "openai/gpt-4.1"
    judge_temperature: float = 0.0
    max_rows: int = 0
    request_delay_seconds: float = 0.0
    retry_attempts: int = 3
    retry_backoff_seconds: float = 10.0


def _default_settings() -> dict[str, object]:
    return {
        "dataset_path": Path("/Users/Anna/Desktop/Pjcts/eval/evaluation_dataset.csv"),
        "output_dir": Path("evaluation/results"),
        "agent_model": "anthropic/claude-sonnet-4-5",
        "judge_model": "openai/gpt-4.1",
        "judge_temperature": 0.0,
        "max_rows": 0,
        "request_delay_seconds": 0.0,
        "retry_attempts": 3,
        "retry_backoff_seconds": 10.0,
    }


def _resolve_path(value: object) -> Path:
    path = value if isinstance(value, Path) else Path(str(value))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _config_path(config_file: str | Path | None) -> Path:
    explicit = config_file or os.getenv("TRAIL_BUDDY_EVAL_CONFIG_FILE")
    if explicit:
        return _resolve_path(explicit)
    return DEFAULT_CONFIG_PATH


def _read_config(config_path: Path) -> dict[str, object]:
    if not config_path.exists():
        return {}
    with config_path.open("rb") as file:
        payload = tomllib.load(file)
    section = payload.get("evaluation", {})
    if not isinstance(section, dict):
        raise ValueError(f"{config_path} must contain an [evaluation] table.")
    return section


def _apply_env_overrides(values: dict[str, object]) -> dict[str, object]:
    env_map = {
        "TRAIL_BUDDY_EVAL_DATASET_PATH": "dataset_path",
        "TRAIL_BUDDY_EVAL_OUTPUT_DIR": "output_dir",
        "TRAIL_BUDDY_EVAL_AGENT_MODEL": "agent_model",
        "TRAIL_BUDDY_EVAL_JUDGE_MODEL": "judge_model",
        "TRAIL_BUDDY_EVAL_MAX_ROWS": "max_rows",
        "TRAIL_BUDDY_EVAL_REQUEST_DELAY_SECONDS": "request_delay_seconds",
        "TRAIL_BUDDY_EVAL_RETRY_ATTEMPTS": "retry_attempts",
        "TRAIL_BUDDY_EVAL_RETRY_BACKOFF_SECONDS": "retry_backoff_seconds",
    }
    merged = dict(values)
    for env_name, field_name in env_map.items():
        env_value = os.getenv(env_name)
        if env_value not in (None, ""):
            merged[field_name] = env_value
    return merged


def load_settings(config_file: str | Path | None = None) -> EvaluationSettings:
    values = _default_settings()
    values.update(_read_config(_config_path(config_file)))
    values = _apply_env_overrides(values)

    return EvaluationSettings(
        dataset_path=_resolve_path(values["dataset_path"]),
        output_dir=_resolve_path(values["output_dir"]),
        agent_model=str(values["agent_model"]),
        judge_model=str(values["judge_model"]),
        judge_temperature=float(values.get("judge_temperature", 0.0)),
        max_rows=int(values.get("max_rows", 0) or 0),
        request_delay_seconds=float(values.get("request_delay_seconds", 0.0) or 0.0),
        retry_attempts=int(values.get("retry_attempts", 3) or 1),
        retry_backoff_seconds=float(values.get("retry_backoff_seconds", 10.0) or 0.0),
    )
