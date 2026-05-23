# Evaluation Pipeline Plan

## Goal

Add an evaluation pipeline that reads cases from:

```text
/Users/Anna/Desktop/Pjcts/eval/evaluation_dataset.csv
```

For each case, the pipeline should send the English query to the real Trail Buddy
agent, compare the generated answer with the ground-truth answer from the CSV when
one exists, and use a separate judge LLM to produce structured evaluation results.
The evaluation should assess two components independently with two separate judge
calls:

1. Truthfulness: whether the assistant answer contains unsupported or
   contradictory claims.
2. Completeness: whether the assistant answer covers the important points from the
   ground-truth answer.

## Proposed Folder Structure

```text
evaluation/
  PLAN.md
  __init__.py
  config.py
  dataset.py
  models.py
  prompts.py
  run.py
  eval_config.toml
  results/
    .gitkeep
```

## Dataset Handling

The current CSV columns are:

```text
query
query (eng)
docs
article_sources
answer
query 2
query 2 (eng)
docs 2
article_sources 2
answer 2
```

Normalize the file into evaluation cases:

- Use `query (eng)` as the primary agent input.
- Use `answer` as the primary ground truth.
- If `query 2 (eng)` is present, create a second case from the same row.
- Use `answer 2` as the second case ground truth.
- If a case has no ground-truth answer, skip it by default or mark it as
  `ungraded` when an explicit CLI flag is passed.

Each normalized case should include:

```text
case_id
source_row
turn
query
ground_truth_answer
has_ground_truth
```

## Configuration

Create `evaluation/eval_config.toml`:

```toml
[evaluation]
dataset_path = "/Users/Anna/Desktop/Pjcts/eval/evaluation_dataset.csv"
output_dir = "evaluation/results"

# LLM used only for judging generated answers.
agent_model = "anthropic/claude-sonnet-4-5"
judge_model = "openai/gpt-4.1"
judge_temperature = 0

# Optional row limit for development runs.
max_rows = 0
```

Environment overrides should be supported:

```text
TRAIL_BUDDY_EVAL_CONFIG_FILE
TRAIL_BUDDY_EVAL_DATASET_PATH
TRAIL_BUDDY_EVAL_OUTPUT_DIR
TRAIL_BUDDY_EVAL_AGENT_MODEL
TRAIL_BUDDY_EVAL_JUDGE_MODEL
TRAIL_BUDDY_EVAL_MAX_ROWS
TRAIL_BUDDY_EVAL_REQUEST_DELAY_SECONDS
TRAIL_BUDDY_EVAL_RETRY_ATTEMPTS
TRAIL_BUDDY_EVAL_RETRY_BACKOFF_SECONDS
```

The Trail Buddy agent model used during evaluation should come from
`agent_model` or `TRAIL_BUDDY_EVAL_AGENT_MODEL`, not from the app-level
`TRAIL_BUDDY_MODEL`.

## Agent Invocation

The runner should call the same graph used by the app:

```python
from langchain_core.messages import HumanMessage

from trail_buddy.graph import build_graph
```

For each case:

1. Build or reuse a Trail Buddy graph.
2. Invoke all turns from the same source row with the same `thread_id`, for
   example `eval-row-{source_row}`, so `query 2` is evaluated as a follow-up
   turn with the earlier row context preserved.
3. Input the English query as a `HumanMessage`.
4. Read the final AI message as `actual_answer`.
5. Store retrieved context from the graph result if available.

Tool use should be configurable later, but the first version can use the default
project behavior so the evaluation reflects the real app path.

## Judge LLM

Use the existing LiteLLM wrapper with a model override:

```python
from trail_buddy.llm import build_llm

judge_llm = build_llm(
    model=settings.judge_model,
    temperature=settings.judge_temperature,
)
```

This keeps provider handling consistent with the rest of the project.

The judge model must be used with structured output. Each judge call should return
only JSON that validates against the relevant Pydantic model. Do not parse
free-form judge prose. Prefer binding the schema directly to the model call, for
example `judge_llm.with_structured_output(TruthfulnessJudgeResult)` and
`judge_llm.with_structured_output(CompletenessJudgeResult)`, or the equivalent
structured-output API supported by the selected provider.

Run two separate judge calls for each graded case:

1. Truthfulness judge call, using a truthfulness-specific system prompt.
2. Completeness judge call, using a completeness-specific system prompt.

## Judge Prompt

Create `evaluation/prompts.py` with two strict system prompts.

Truthfulness system prompt:

```text
You are a Truthfulness evaluation judge for Trail Buddy.

Compare the assistant answer against the ground-truth answer.
Evaluate only against the ground truth. Do not reward extra information unless it is
consistent with the ground truth. Penalize contradictions, unsupported claims,
invented facts, and unsafe advice.

Metric:
- truthfulness: true only if the assistant answer contains no claim that conflicts
  with or is unsupported by the ground-truth answer.

Return structured output only.
```

Completeness system prompt:

```text
You are a Completeness evaluation judge for Trail Buddy.

Compare the assistant answer against the ground-truth answer.
Evaluate only whether the assistant answer includes all important information from
the ground-truth answer. Do not penalize extra information here unless it replaces,
obscures, or contradicts a required ground-truth point. Missing key facts should be
penalized.

Metric:
- completeness: true only if the assistant answer covers all important
  ground-truth points.

Return structured output only.
```

The user prompt to each judge should include:

```text
Query:
...

Ground-truth answer:
...

Assistant answer:
...
```

## Structured Output

Create `evaluation/models.py` with Pydantic models:

```python
from pydantic import BaseModel, Field


class TruthfulnessJudgeResult(BaseModel):
    truthfulness: bool = Field(description="No unsupported or contradictory claims.")
    explanation: str
    unsupported_claims: list[str]
    contradictory_claims: list[str]


class CompletenessJudgeResult(BaseModel):
    completeness: bool = Field(description="All important ground-truth points are covered.")
    explanation: str
    missing_ground_truth_points: list[str]


class JudgeResult(BaseModel):
    truthfulness: TruthfulnessJudgeResult
    completeness: CompletenessJudgeResult
    overall_pass: bool
```

`overall_pass` should be
`truthfulness.truthfulness and completeness.completeness`.

## Runner CLI

Create `evaluation/run.py` and support:

```bash
uv run python -m evaluation.run
uv run python -m evaluation.run --limit 5
uv run python -m evaluation.run --include-ungraded
uv run python -m evaluation.run --output evaluation/results/test.jsonl
```

The runner should:

1. Load config.
2. Load and normalize CSV cases.
3. Invoke Trail Buddy for each case.
4. Skip judge step when no ground truth exists unless `--include-ungraded` is set.
5. Send query, ground truth, and assistant answer to the Truthfulness judge.
6. Validate the Truthfulness judge response with `TruthfulnessJudgeResult`.
7. Send query, ground truth, and assistant answer to the Completeness judge.
8. Validate the Completeness judge response with `CompletenessJudgeResult`.
9. Combine both structured results into `JudgeResult`.
10. Write JSONL result rows and a summary JSON file.

## Result Files

Write results into `evaluation/results/`:

```text
eval_YYYYMMDD_HHMMSS.jsonl
eval_YYYYMMDD_HHMMSS_summary.json
```

Each JSONL row should contain:

```json
{
  "case_id": "row-1-turn-1",
  "source_row": 1,
  "turn": 1,
  "query": "...",
  "ground_truth_answer": "...",
  "actual_answer": "...",
  "retrieved": [],
  "judgement": {
    "truthfulness": {
      "truthfulness": true,
      "explanation": "...",
      "unsupported_claims": [],
      "contradictory_claims": []
    },
    "completeness": {
      "completeness": false,
      "explanation": "...",
      "missing_ground_truth_points": ["..."]
    },
    "overall_pass": false
  }
}
```

The summary should include:

```text
total_cases
graded_cases
ungraded_cases
truthfulness_pass_rate
completeness_pass_rate
overall_pass_rate
output_path
```

## Tests

Add `tests/test_evaluation.py` with fake LLMs and no API keys:

- CSV rows normalize into one or two cases correctly.
- Empty ground-truth answers are skipped by default.
- Structured judge output validates through `TruthfulnessJudgeResult`,
  `CompletenessJudgeResult`, and the combined `JudgeResult`.
- End-to-end runner works with a fake agent answer and fake judge result.
- Summary counts and pass rates are computed correctly.

## Implementation Order

1. Add config and data models.
2. Add CSV normalization.
3. Add agent invocation helper.
4. Add two judge prompts and structured output parsing.
5. Add JSONL and summary writing.
6. Add CLI arguments.
7. Add tests.
8. Document the command in `README.md`.
