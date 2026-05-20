from __future__ import annotations

from dataclasses import dataclass

from examples.aime_math.config import _env_bool, _env_float, _env_int


@dataclass(frozen=True)
class ParentReflectionGEPAConfig:
    candidate_pool_size: int = 5
    reflect_train_batch_size: int = 4
    probe_val_batch_size: int = 4
    max_outer_steps: int = 32
    score_sampling_temperature: float = 1.0
    num_parallel_branches: int = 4
    ucb_exploration_coef: float = 0.25
    prompt_ucb_exploration_coef: float = 0.25
    prompt_score_std_floor: float = 1e-3
    train_rejection_max_retries: int = 8
    val_rejection_max_retries: int = 8
    use_parent_history: bool = True

    def __post_init__(self) -> None:
        if self.candidate_pool_size < 1:
            raise ValueError("candidate_pool_size must be >= 1")
        if self.reflect_train_batch_size < 1:
            raise ValueError("reflect_train_batch_size must be >= 1")
        if self.probe_val_batch_size < 0:
            raise ValueError("probe_val_batch_size must be >= 0")
        if self.max_outer_steps < 1:
            raise ValueError("max_outer_steps must be >= 1")
        if self.score_sampling_temperature <= 0:
            raise ValueError("score_sampling_temperature must be > 0")
        if self.num_parallel_branches < 1:
            raise ValueError("num_parallel_branches must be >= 1")
        if self.ucb_exploration_coef < 0:
            raise ValueError("ucb_exploration_coef must be >= 0")
        if self.prompt_ucb_exploration_coef < 0:
            raise ValueError("prompt_ucb_exploration_coef must be >= 0")
        if self.prompt_score_std_floor <= 0:
            raise ValueError("prompt_score_std_floor must be > 0")
        if self.train_rejection_max_retries < 1:
            raise ValueError("train_rejection_max_retries must be >= 1")
        if self.val_rejection_max_retries < 1:
            raise ValueError("val_rejection_max_retries must be >= 1")

    @classmethod
    def from_env(cls) -> "ParentReflectionGEPAConfig":
        return cls(
            candidate_pool_size=_env_int("AIME_PRG_CANDIDATE_POOL_SIZE", 5),
            reflect_train_batch_size=_env_int("AIME_PRG_REFLECT_TRAIN_BATCH_SIZE", 4),
            probe_val_batch_size=_env_int("AIME_PRG_PROBE_VAL_BATCH_SIZE", 4),
            max_outer_steps=_env_int("AIME_PRG_MAX_OUTER_STEPS", 32),
            score_sampling_temperature=_env_float("AIME_PRG_SCORE_SAMPLING_TEMPERATURE", 1.0),
            num_parallel_branches=_env_int("AIME_PRG_NUM_PARALLEL_BRANCHES", 4),
            ucb_exploration_coef=_env_float("AIME_PRG_UCB_EXPLORATION_COEF", 0.25),
            prompt_ucb_exploration_coef=_env_float("AIME_PRG_PROMPT_UCB_EXPLORATION_COEF", 0.25),
            prompt_score_std_floor=_env_float("AIME_PRG_PROMPT_SCORE_STD_FLOOR", 1e-3),
            train_rejection_max_retries=_env_int("AIME_PRG_TRAIN_REJECTION_MAX_RETRIES", 8),
            val_rejection_max_retries=_env_int("AIME_PRG_VAL_REJECTION_MAX_RETRIES", 8),
            use_parent_history=_env_bool("AIME_PRG_USE_PARENT_HISTORY", True),
        )
