from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping
from pathlib import Path

from evaluation.models import EvaluationCase


PRIMARY_QUERY_COL = "query (eng)"
PRIMARY_ANSWER_COL = "answer"
SECONDARY_QUERY_COL = "query 2 (eng)"
SECONDARY_ANSWER_COL = "answer 2"


def _clean(value: object) -> str:
    return str(value or "").strip()


def _case_from_row(
    row: Mapping[str, object],
    *,
    source_row: int,
    turn: int,
    query_col: str,
    answer_col: str,
) -> EvaluationCase | None:
    query = _clean(row.get(query_col))
    if not query:
        return None

    ground_truth = _clean(row.get(answer_col))
    return EvaluationCase(
        case_id=f"row-{source_row}-turn-{turn}",
        source_row=source_row,
        turn=turn,
        query=query,
        ground_truth_answer=ground_truth,
        has_ground_truth=bool(ground_truth),
    )


def normalize_rows(
    rows: Iterable[Mapping[str, object]],
    *,
    include_ungraded: bool = False,
) -> list[EvaluationCase]:
    cases: list[EvaluationCase] = []
    for source_row, row in enumerate(rows, start=1):
        for turn, query_col, answer_col in (
            (1, PRIMARY_QUERY_COL, PRIMARY_ANSWER_COL),
            (2, SECONDARY_QUERY_COL, SECONDARY_ANSWER_COL),
        ):
            case = _case_from_row(
                row,
                source_row=source_row,
                turn=turn,
                query_col=query_col,
                answer_col=answer_col,
            )
            if case is None:
                continue
            if not case.has_ground_truth and not include_ungraded:
                continue
            cases.append(case)
    return cases


def load_cases(
    dataset_path: str | Path,
    *,
    max_rows: int = 0,
    include_ungraded: bool = False,
) -> list[EvaluationCase]:
    path = Path(dataset_path)
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row_number, row in enumerate(reader, start=1):
            if max_rows > 0 and row_number > max_rows:
                break
            rows.append(row)
    return normalize_rows(rows, include_ungraded=include_ungraded)
