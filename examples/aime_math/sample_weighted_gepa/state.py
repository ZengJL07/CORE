from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CandidateRecord:
    candidate_id: int
    prompt: str
    val_score: float
    parent_id: int | None
    train_scores_by_sample: dict[str, float]


@dataclass(frozen=True)
class InnerLoopAttempt:
    step_idx: int
    prompt: str
    reflection: str
    train_score: float
    mixed_score: float | None = None


@dataclass(frozen=True)
class OuterStepRecord:
    outer_step: int
    base_candidate_id: int
    base_prompt: str
    reflect_train_ids: list[str]
    probe_val_ids: list[str]
    base_mixed_score: float
    candidate_mixed_score: float | None
    candidate_val_score: float | None
    accepted_to_pool: bool
    total_metric_calls: int
    inner_attempts: list[InnerLoopAttempt] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SampleWeightedGEPARunResult:
    best_prompt: str
    best_val_score: float
    total_metric_calls: int
    total_outer_steps: int
    candidate_pool: list[CandidateRecord]
    outer_step_records: list[OuterStepRecord]
