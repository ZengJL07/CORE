from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrajectoryRecord:
    trajectory_id: str
    input_text: str
    gold_answer: str
    model_answer: str
    reasoning: str
    score: float
    feedback: str

    @property
    def is_success(self) -> bool:
        return self.score >= 1.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class AnalystSuggestion:
    trajectory_id: str
    analyst_kind: str
    source_score: float
    suggestion: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class Trace2SkillBaselineResult:
    optimized_prompt: str
    val_score: float
    metric_calls_used: int
    num_trajectories: int
    num_suggestions: int
    num_iterations: int
    output_dir: Path
