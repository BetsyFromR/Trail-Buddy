from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EvaluationCase(BaseModel):
    case_id: str
    source_row: int
    turn: int
    query: str
    ground_truth_answer: str = ""
    has_ground_truth: bool


class AgentRunResult(BaseModel):
    actual_answer: str
    retrieved: list[str] = Field(default_factory=list)


class TruthfulnessJudgeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    truthfulness: bool = Field(description="No unsupported or contradictory claims.")
    explanation: str
    unsupported_claims: list[str] = Field(default_factory=list)
    contradictory_claims: list[str] = Field(default_factory=list)


class CompletenessJudgeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    completeness: bool = Field(
        description="All important ground-truth points are covered."
    )
    explanation: str
    missing_ground_truth_points: list[str] = Field(default_factory=list)


class JudgeResult(BaseModel):
    truthfulness: TruthfulnessJudgeResult
    completeness: CompletenessJudgeResult
    overall_pass: bool

    @model_validator(mode="after")
    def _overall_pass_matches_components(self) -> JudgeResult:
        expected = self.truthfulness.truthfulness and self.completeness.completeness
        if self.overall_pass != expected:
            raise ValueError("overall_pass must equal truthfulness and completeness.")
        return self

    @classmethod
    def from_components(
        cls,
        *,
        truthfulness: TruthfulnessJudgeResult,
        completeness: CompletenessJudgeResult,
    ) -> JudgeResult:
        return cls(
            truthfulness=truthfulness,
            completeness=completeness,
            overall_pass=truthfulness.truthfulness and completeness.completeness,
        )


class EvaluationResult(BaseModel):
    case_id: str
    source_row: int
    turn: int
    query: str
    ground_truth_answer: str
    actual_answer: str
    retrieved: list[str] = Field(default_factory=list)
    judgement: JudgeResult | None = None


class EvaluationSummary(BaseModel):
    total_cases: int
    graded_cases: int
    ungraded_cases: int
    truthfulness_pass_rate: float
    completeness_pass_rate: float
    overall_pass_rate: float
    output_path: str


class EvaluationRunOutput(BaseModel):
    output_path: Path
    summary_path: Path
    summary: EvaluationSummary
    results: list[EvaluationResult]

    model_config = ConfigDict(arbitrary_types_allowed=True)
