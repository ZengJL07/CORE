from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CandidateRecord:
    candidate_id: int
    prompt: str
    val_score: float
    parent_id: int | None
    parent_prompt: str | None
    reflection_from_parent: str
    val_scores_by_sample: dict[str, float]
    train_scores_by_sample: dict[str, float] = field(default_factory=dict)
    times_selected_as_current: int = 0
    val_sampled_count: dict[str, int] = field(default_factory=dict)


@dataclass
class BranchRecord:
    branch_idx: int
    current_candidate_id: int
    current_prompt: str
    parent_prompt_used: str | None
    parent_reflection_used: str
    train_batch_ids: list[str]
    probe_val_ids: list[str]
    base_train_score: float
    base_total_score: float
    generated_reflection: str
    final_prompt: str
    train_score: float
    total_score: float | None
    passed_train_gate: bool
    passed_total_gate: bool
    full_val_score: float | None = None
    accepted_to_pool: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OuterStepRecord:
    outer_step: int
    sampled_current_candidate_ids: list[int]
    inserted_candidate_ids: list[int]
    branch_records: list[BranchRecord] = field(default_factory=list)
    total_metric_calls: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidatePoint:
    iteration: int
    outer_step: int
    branch_idx: int
    candidate_id: int
    score: float
    accepted_to_pool: bool


@dataclass
class ParentReflectionGEPARunResult:
    best_prompt: str
    best_val_score: float
    total_metric_calls: int
    total_outer_steps: int
    candidate_pool: list[CandidateRecord]
    outer_step_records: list[OuterStepRecord]
    candidate_points: list[CandidatePoint]
