from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from evaluation.models import EvaluationCase


TRUTHFULNESS_SYSTEM_PROMPT = """You are a Truthfulness evaluation judge for Trail Buddy.

Compare the assistant answer against the ground-truth answer.
Evaluate only against the ground truth. Do not reward extra information unless it is
consistent with the ground truth. Penalize contradictions, unsupported claims,
invented facts, and unsafe advice.

Metric:
- truthfulness: true only if the assistant answer contains no claim that conflicts
  with or is unsupported by the ground-truth answer.

Return structured output only."""


COMPLETENESS_SYSTEM_PROMPT = """You are a Completeness evaluation judge for Trail Buddy.

Compare the assistant answer against the ground-truth answer.
Evaluate only whether the assistant answer includes all important information from
the ground-truth answer. Do not penalize extra information here unless it replaces,
obscures, or contradicts a required ground-truth point. Missing key facts should be
penalized.

Metric:
- completeness: true only if the assistant answer covers all important
  ground-truth points.

Return structured output only."""


def build_judge_user_prompt(case: EvaluationCase, actual_answer: str) -> str:
    return "\n\n".join(
        [
            "Query:\n" + case.query,
            "Ground-truth answer:\n" + case.ground_truth_answer,
            "Assistant answer:\n" + actual_answer,
        ]
    )


def build_truthfulness_messages(case: EvaluationCase, actual_answer: str):
    return [
        SystemMessage(content=TRUTHFULNESS_SYSTEM_PROMPT),
        HumanMessage(content=build_judge_user_prompt(case, actual_answer)),
    ]

def build_completeness_messages(case: EvaluationCase, actual_answer: str):
    return [
        SystemMessage(content=COMPLETENESS_SYSTEM_PROMPT),
        HumanMessage(content=build_judge_user_prompt(case, actual_answer)),
    ]
