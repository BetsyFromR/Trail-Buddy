import csv
import json

import pytest
from langchain_core.messages import AIMessage

from evaluation.config import EvaluationSettings, load_settings
from evaluation.dataset import normalize_rows
from evaluation.models import (
    AgentRunResult,
    CompletenessJudgeResult,
    EvaluationCase,
    EvaluationResult,
    JudgeResult,
    TruthfulnessJudgeResult,
)
from evaluation.run import (
    _call_with_retry,
    build_judge_evaluator,
    invoke_agent,
    run_evaluation,
    summarize_results,
)


def _write_dataset(path, rows):
    fieldnames = [
        "query",
        "query (eng)",
        "docs",
        "article_sources",
        "answer",
        "query 2",
        "query 2 (eng)",
        "docs 2",
        "article_sources 2",
        "answer 2",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _judge_result(
    *,
    truthfulness: bool = True,
    completeness: bool = True,
) -> JudgeResult:
    return JudgeResult.from_components(
        truthfulness=TruthfulnessJudgeResult(
            truthfulness=truthfulness,
            explanation="truthfulness explanation",
            unsupported_claims=[] if truthfulness else ["unsupported detail"],
            contradictory_claims=[],
        ),
        completeness=CompletenessJudgeResult(
            completeness=completeness,
            explanation="completeness explanation",
            missing_ground_truth_points=[] if completeness else ["required detail"],
        ),
    )


def test_evaluation_agent_model_is_separate_from_app_model(monkeypatch, tmp_path):
    config_path = tmp_path / "eval_config.toml"
    config_path.write_text(
        "\n".join(
            [
                "[evaluation]",
                f'dataset_path = "{tmp_path / "dataset.csv"}"',
                'output_dir = "evaluation/results"',
                'agent_model = "anthropic/claude-sonnet-4-5"',
                'judge_model = "openai/gpt-4.1"',
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("TRAIL_BUDDY_MODEL", "openai/gpt-4.1")

    settings = load_settings(config_path)

    assert settings.agent_model == "anthropic/claude-sonnet-4-5"

    monkeypatch.setenv("TRAIL_BUDDY_EVAL_AGENT_MODEL", "mistral/mistral-large-latest")
    overridden = load_settings(config_path)
    assert overridden.agent_model == "mistral/mistral-large-latest"


def test_csv_rows_normalize_into_one_or_two_cases_correctly():
    cases = normalize_rows(
        [
            {
                "query (eng)": "How should I pace a 35K trail race?",
                "answer": "Start conservatively.",
                "query 2 (eng)": "What should I eat?",
                "answer 2": "Eat early and often.",
            },
            {
                "query (eng)": "Which shoes should I wear?",
                "answer": "Use trail shoes.",
                "query 2 (eng)": "",
                "answer 2": "",
            },
        ]
    )

    assert [case.case_id for case in cases] == [
        "row-1-turn-1",
        "row-1-turn-2",
        "row-2-turn-1",
    ]
    assert cases[0].query == "How should I pace a 35K trail race?"
    assert cases[1].turn == 2
    assert cases[1].ground_truth_answer == "Eat early and often."


def test_empty_ground_truth_answers_are_skipped_by_default():
    rows = [{"query (eng)": "Can I run this race?", "answer": ""}]

    assert normalize_rows(rows) == []

    cases = normalize_rows(rows, include_ungraded=True)
    assert len(cases) == 1
    assert cases[0].has_ground_truth is False
    assert cases[0].ground_truth_answer == ""


def test_structured_judge_output_validates_through_component_and_combined_models():
    truthfulness = TruthfulnessJudgeResult(
        truthfulness=True,
        explanation="No unsupported claims.",
        unsupported_claims=[],
        contradictory_claims=[],
    )
    completeness = CompletenessJudgeResult(
        completeness=False,
        explanation="Missing the required pacing point.",
        missing_ground_truth_points=["Start conservatively."],
    )

    result = JudgeResult.from_components(
        truthfulness=truthfulness,
        completeness=completeness,
    )

    assert result.truthfulness.truthfulness is True
    assert result.completeness.completeness is False
    assert result.overall_pass is False


def test_combined_judge_result_rejects_incorrect_overall_pass():
    with pytest.raises(ValueError, match="overall_pass"):
        JudgeResult(
            truthfulness=TruthfulnessJudgeResult(
                truthfulness=True,
                explanation="ok",
            ),
            completeness=CompletenessJudgeResult(
                completeness=False,
                explanation="missing",
                missing_ground_truth_points=["point"],
            ),
            overall_pass=True,
        )


def test_judge_evaluator_uses_two_structured_calls_with_separate_prompts():
    calls = []

    class FakeStructuredJudge:
        def __init__(self, schema):
            self.schema = schema

        def invoke(self, messages):
            calls.append((self.schema, messages))
            if self.schema is TruthfulnessJudgeResult:
                return TruthfulnessJudgeResult(
                    truthfulness=True,
                    explanation="truthful",
                )
            return CompletenessJudgeResult(
                completeness=False,
                explanation="missing a point",
                missing_ground_truth_points=["point"],
            )

    class FakeJudgeLLM:
        def __init__(self):
            self.schemas = []

        def with_structured_output(self, schema):
            self.schemas.append(schema)
            return FakeStructuredJudge(schema)

    fake_llm = FakeJudgeLLM()
    evaluator = build_judge_evaluator(fake_llm)
    case = EvaluationCase(
        case_id="row-1-turn-1",
        source_row=1,
        turn=1,
        query="How should I pace?",
        ground_truth_answer="Start slowly.",
        has_ground_truth=True,
    )

    result = evaluator(case, "Start fast.")

    assert fake_llm.schemas == [TruthfulnessJudgeResult, CompletenessJudgeResult]
    assert len(calls) == 2
    assert "Truthfulness evaluation judge" in calls[0][1][0].content
    assert "Completeness evaluation judge" in calls[1][1][0].content
    assert result.truthfulness.truthfulness is True
    assert result.completeness.completeness is False
    assert result.overall_pass is False


def test_agent_invocation_preserves_thread_per_source_row():
    class FakeGraph:
        def __init__(self):
            self.messages_by_thread = {}
            self.thread_ids = []

        def invoke(self, inputs, config):
            thread_id = config["configurable"]["thread_id"]
            self.thread_ids.append(thread_id)
            messages = self.messages_by_thread.setdefault(thread_id, [])
            messages.extend(inputs["messages"])
            user_turns = [
                message for message in messages if getattr(message, "type", None) == "human"
            ]
            messages.append(AIMessage(content=f"seen {len(user_turns)} user turns"))
            return {"messages": list(messages), "retrieved": []}

    graph = FakeGraph()
    first = EvaluationCase(
        case_id="row-1-turn-1",
        source_row=1,
        turn=1,
        query="Can I run this race?",
        ground_truth_answer="Maybe.",
        has_ground_truth=True,
    )
    second = EvaluationCase(
        case_id="row-1-turn-2",
        source_row=1,
        turn=2,
        query="It is Boka Bay Blue.",
        ground_truth_answer="Estimate 6-7 hours.",
        has_ground_truth=True,
    )

    first_result = invoke_agent(first, graph)
    second_result = invoke_agent(second, graph)

    assert graph.thread_ids == ["eval-row-1", "eval-row-1"]
    assert first_result.actual_answer == "seen 1 user turns"
    assert second_result.actual_answer == "seen 2 user turns"


def test_retry_helper_retries_rate_limit_errors():
    calls = []
    sleeps = []

    def flaky_call():
        calls.append("call")
        if len(calls) == 1:
            raise RuntimeError("429 Too Many Requests")
        return "ok"

    result = _call_with_retry(
        flaky_call,
        label="fake call",
        retry_attempts=2,
        retry_backoff_seconds=3,
        sleeper=sleeps.append,
    )

    assert result == "ok"
    assert calls == ["call", "call"]
    assert sleeps == [3]


def test_retry_helper_does_not_retry_insufficient_quota():
    calls = []
    sleeps = []

    def quota_error():
        calls.append("call")
        raise RuntimeError(
            "429 insufficient_quota: You exceeded your current quota, "
            "please check your plan and billing details."
        )

    with pytest.raises(RuntimeError, match="insufficient_quota"):
        _call_with_retry(
            quota_error,
            label="fake call",
            retry_attempts=5,
            retry_backoff_seconds=3,
            sleeper=sleeps.append,
        )

    assert calls == ["call"]
    assert sleeps == []


def test_end_to_end_runner_works_with_fake_agent_and_fake_judge(tmp_path):
    dataset_path = tmp_path / "evaluation_dataset.csv"
    _write_dataset(
        dataset_path,
        [
            {
                "query (eng)": "How should I pace?",
                "answer": "Start conservatively.",
                "query 2 (eng)": "What should I eat?",
                "answer 2": "Eat early.",
            }
        ],
    )
    settings = EvaluationSettings(
        dataset_path=dataset_path,
        output_dir=tmp_path / "results",
        judge_model="fake",
    )

    def fake_agent(case):
        return AgentRunResult(
            actual_answer=f"Actual answer for {case.case_id}",
            retrieved=[f"Retrieved for {case.query}"],
        )

    def fake_judge(case, actual_answer):
        return _judge_result(
            truthfulness=True,
            completeness=case.turn == 1,
        )

    output_path = tmp_path / "results" / "test.jsonl"
    output = run_evaluation(
        settings,
        output_path=output_path,
        agent_invoker=fake_agent,
        judge_evaluator=fake_judge,
    )

    assert output.summary.total_cases == 2
    assert output.summary.graded_cases == 2
    assert output.summary.truthfulness_pass_rate == 1.0
    assert output.summary.completeness_pass_rate == 0.5
    assert output.summary.overall_pass_rate == 0.5
    assert output.summary_path == tmp_path / "results" / "test_summary.json"

    rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0]["case_id"] == "row-1-turn-1"
    assert rows[0]["retrieved"] == ["Retrieved for How should I pace?"]
    assert rows[1]["judgement"]["completeness"]["completeness"] is False


def test_summary_counts_and_pass_rates_are_computed_correctly(tmp_path):
    results = [
        EvaluationResult(
            case_id="row-1-turn-1",
            source_row=1,
            turn=1,
            query="q1",
            ground_truth_answer="a1",
            actual_answer="actual 1",
            judgement=_judge_result(truthfulness=True, completeness=True),
        ),
        EvaluationResult(
            case_id="row-2-turn-1",
            source_row=2,
            turn=1,
            query="q2",
            ground_truth_answer="a2",
            actual_answer="actual 2",
            judgement=_judge_result(truthfulness=True, completeness=False),
        ),
        EvaluationResult(
            case_id="row-3-turn-1",
            source_row=3,
            turn=1,
            query="q3",
            ground_truth_answer="",
            actual_answer="actual 3",
            judgement=None,
        ),
    ]

    summary = summarize_results(results, output_path=tmp_path / "eval.jsonl")

    assert summary.total_cases == 3
    assert summary.graded_cases == 2
    assert summary.ungraded_cases == 1
    assert summary.truthfulness_pass_rate == 1.0
    assert summary.completeness_pass_rate == 0.5
    assert summary.overall_pass_rate == 0.5
