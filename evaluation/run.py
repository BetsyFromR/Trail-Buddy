from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from evaluation.config import EvaluationSettings, load_settings
from evaluation.dataset import load_cases
from evaluation.models import (
    AgentRunResult,
    CompletenessJudgeResult,
    EvaluationCase,
    EvaluationResult,
    EvaluationRunOutput,
    EvaluationSummary,
    JudgeResult,
    TruthfulnessJudgeResult,
)
from evaluation.prompts import (
    build_completeness_messages,
    build_truthfulness_messages,
)
from trail_buddy.graph import build_graph
from trail_buddy.llm import build_llm
from trail_buddy.logging_config import configure_logging


AgentInvoker = Callable[[EvaluationCase], AgentRunResult]
JudgeEvaluator = Callable[[EvaluationCase, str], JudgeResult]
logger = logging.getLogger(__name__)


def _message_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts)
    return str(content or "")


def invoke_agent(case: EvaluationCase, graph) -> AgentRunResult:
    result = graph.invoke(
        {"messages": [HumanMessage(content=case.query)]},
        config={"configurable": {"thread_id": f"eval-row-{case.source_row}"}},
    )
    messages = result.get("messages", [])
    actual_answer = _message_text(getattr(messages[-1], "content", "")) if messages else ""
    return AgentRunResult(
        actual_answer=actual_answer,
        retrieved=list(result.get("retrieved") or []),
    )


def _is_rate_limit_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    if (
        "insufficient_quota" in text
        or "exceeded your current quota" in text
        or "check your plan and billing" in text
    ):
        return False

    response = getattr(exc, "response", None)
    status_code = getattr(exc, "status_code", None) or getattr(
        response,
        "status_code",
        None,
    )
    if status_code == 429:
        return True

    return (
        "429" in text
        or "rate limit" in text
        or "rate_limited" in text
        or "too many requests" in text
    )


def _call_with_retry(
    call: Callable[[], object],
    *,
    label: str,
    retry_attempts: int,
    retry_backoff_seconds: float,
    sleeper: Callable[[float], None] = time.sleep,
):
    attempts = max(1, retry_attempts)
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except Exception as exc:
            if attempt >= attempts or not _is_rate_limit_error(exc):
                raise

            wait_seconds = max(0.0, retry_backoff_seconds) * attempt
            logger.warning(
                "[eval] %s rate limited; retrying in %.1fs (%s/%s)",
                label,
                wait_seconds,
                attempt + 1,
                attempts,
            )
            sleeper(wait_seconds)


def _sleep_before_llm_call(
    request_delay_seconds: float,
    *,
    sleeper: Callable[[float], None] = time.sleep,
) -> None:
    if request_delay_seconds > 0:
        sleeper(request_delay_seconds)


def _coerce_structured_result(result, model):
    if isinstance(result, model):
        return result
    if isinstance(result, dict):
        return model.model_validate(result)
    raise TypeError(
        f"Structured judge returned {type(result).__name__}; expected {model.__name__}."
    )


def build_judge_evaluator(
    judge_llm,
    *,
    request_delay_seconds: float = 0.0,
    retry_attempts: int = 1,
    retry_backoff_seconds: float = 0.0,
    sleeper: Callable[[float], None] = time.sleep,
) -> JudgeEvaluator:
    try:
        truthfulness_judge = judge_llm.with_structured_output(TruthfulnessJudgeResult)
        completeness_judge = judge_llm.with_structured_output(CompletenessJudgeResult)
    except AttributeError as exc:
        raise TypeError("judge_llm must support with_structured_output().") from exc

    def _judge(case: EvaluationCase, actual_answer: str) -> JudgeResult:
        _sleep_before_llm_call(request_delay_seconds, sleeper=sleeper)
        truthfulness = _coerce_structured_result(
            _call_with_retry(
                lambda: truthfulness_judge.invoke(
                    build_truthfulness_messages(case, actual_answer)
                ),
                label=f"truthfulness judge for {case.case_id}",
                retry_attempts=retry_attempts,
                retry_backoff_seconds=retry_backoff_seconds,
                sleeper=sleeper,
            ),
            TruthfulnessJudgeResult,
        )
        _sleep_before_llm_call(request_delay_seconds, sleeper=sleeper)
        completeness = _coerce_structured_result(
            _call_with_retry(
                lambda: completeness_judge.invoke(
                    build_completeness_messages(case, actual_answer)
                ),
                label=f"completeness judge for {case.case_id}",
                retry_attempts=retry_attempts,
                retry_backoff_seconds=retry_backoff_seconds,
                sleeper=sleeper,
            ),
            CompletenessJudgeResult,
        )
        return JudgeResult.from_components(
            truthfulness=truthfulness,
            completeness=completeness,
        )

    return _judge


def build_agent_invoker(
    graph,
    settings: EvaluationSettings,
    *,
    sleeper: Callable[[float], None] = time.sleep,
) -> AgentInvoker:
    def _invoke(case: EvaluationCase) -> AgentRunResult:
        _sleep_before_llm_call(settings.request_delay_seconds, sleeper=sleeper)
        return _call_with_retry(
            lambda: invoke_agent(case, graph),
            label=f"agent for {case.case_id}",
            retry_attempts=settings.retry_attempts,
            retry_backoff_seconds=settings.retry_backoff_seconds,
            sleeper=sleeper,
        )

    return _invoke


def _default_output_path(output_dir: Path, now: datetime | None = None) -> Path:
    timestamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return output_dir / f"eval_{timestamp}.jsonl"


def _summary_path(output_path: Path) -> Path:
    if output_path.suffix == ".jsonl":
        return output_path.with_name(f"{output_path.stem}_summary.json")
    return output_path.with_name(f"{output_path.name}_summary.json")


def summarize_results(
    results: Sequence[EvaluationResult],
    *,
    output_path: Path,
) -> EvaluationSummary:
    graded = [result for result in results if result.judgement is not None]
    graded_count = len(graded)

    def pass_rate(predicate: Callable[[JudgeResult], bool]) -> float:
        if graded_count == 0:
            return 0.0
        passed = sum(1 for result in graded if predicate(result.judgement))
        return passed / graded_count

    return EvaluationSummary(
        total_cases=len(results),
        graded_cases=graded_count,
        ungraded_cases=len(results) - graded_count,
        truthfulness_pass_rate=pass_rate(
            lambda judgement: judgement.truthfulness.truthfulness
        ),
        completeness_pass_rate=pass_rate(
            lambda judgement: judgement.completeness.completeness
        ),
        overall_pass_rate=pass_rate(lambda judgement: judgement.overall_pass),
        output_path=str(output_path),
    )


def write_results(
    results: Sequence[EvaluationResult],
    *,
    output_path: Path,
) -> EvaluationSummary:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for result in results:
            payload = result.model_dump(mode="json")
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")

    summary = summarize_results(results, output_path=output_path)
    summary_path = _summary_path(output_path)
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary.model_dump(mode="json"), file, ensure_ascii=False, indent=2)
        file.write("\n")
    return summary


def run_evaluation(
    settings: EvaluationSettings,
    *,
    limit: int | None = None,
    include_ungraded: bool = False,
    output_path: str | Path | None = None,
    agent_invoker: AgentInvoker | None = None,
    judge_evaluator: JudgeEvaluator | None = None,
    now: datetime | None = None,
) -> EvaluationRunOutput:
    row_limit = settings.max_rows if limit is None else limit
    cases = load_cases(
        settings.dataset_path,
        max_rows=row_limit,
        include_ungraded=include_ungraded,
    )

    resolved_output_path = (
        Path(output_path)
        if output_path is not None
        else _default_output_path(settings.output_dir, now=now)
    )

    if agent_invoker is None:
        agent_llm = build_llm(model=settings.agent_model)
        graph = build_graph(llm=agent_llm)
        agent_invoker = build_agent_invoker(graph, settings)

    if judge_evaluator is None:
        judge_llm = build_llm(
            model=settings.judge_model,
            temperature=settings.judge_temperature,
        )
        judge_evaluator = build_judge_evaluator(
            judge_llm,
            request_delay_seconds=settings.request_delay_seconds,
            retry_attempts=settings.retry_attempts,
            retry_backoff_seconds=settings.retry_backoff_seconds,
        )

    results: list[EvaluationResult] = []
    for case in cases:
        agent_result = agent_invoker(case)
        judgement = None
        if case.has_ground_truth:
            judgement = judge_evaluator(case, agent_result.actual_answer)

        results.append(
            EvaluationResult(
                case_id=case.case_id,
                source_row=case.source_row,
                turn=case.turn,
                query=case.query,
                ground_truth_answer=case.ground_truth_answer,
                actual_answer=agent_result.actual_answer,
                retrieved=agent_result.retrieved,
                judgement=judgement,
            )
        )

    summary = write_results(results, output_path=resolved_output_path)
    return EvaluationRunOutput(
        output_path=resolved_output_path,
        summary_path=_summary_path(resolved_output_path),
        summary=summary,
        results=results,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Trail Buddy evaluation.")
    parser.add_argument(
        "--config",
        help="Path to an evaluation config file. Defaults to evaluation/eval_config.toml.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of source CSV rows to load for this run.",
    )
    parser.add_argument(
        "--include-ungraded",
        action="store_true",
        help="Include cases with no ground-truth answer and skip judging them.",
    )
    parser.add_argument(
        "--output",
        help="JSONL output path. Defaults to evaluation/results/eval_TIMESTAMP.jsonl.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv()
    configure_logging()
    args = parse_args(argv)
    settings = load_settings(args.config)
    output = run_evaluation(
        settings,
        limit=args.limit,
        include_ungraded=args.include_ungraded,
        output_path=args.output,
    )
    print(f"Wrote evaluation results to {output.output_path}")
    print(f"Wrote evaluation summary to {output.summary_path}")
    print(output.summary.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
