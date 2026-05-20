from __future__ import annotations

import os
from dataclasses import dataclass

from examples.aime_math.config import _env_float, _env_int


@dataclass(frozen=True)
class SampleWeightedGEPAConfig:
    candidate_pool_size: int = 5
    reflect_train_batch_size: int = 4
    probe_val_batch_size: int = 4
    inner_steps: int = 3
    max_outer_steps: int = 32
    all_fail_bonus: float = 0.1
    score_sampling_temperature: float = 1.0
    num_parallel_branches: int = 4
    ucb_exploration_coef: float = 0.25
    train_rejection_max_retries: int = 8
    val_rejection_max_retries: int = 8

    def __post_init__(self) -> None:
        if self.candidate_pool_size < 1:
            raise ValueError("candidate_pool_size must be >= 1")
        if self.reflect_train_batch_size < 1:
            raise ValueError("reflect_train_batch_size must be >= 1")
        if self.probe_val_batch_size < 1:
            raise ValueError("probe_val_batch_size must be >= 1")
        if self.inner_steps < 1:
            raise ValueError("inner_steps must be >= 1")
        if self.max_outer_steps < 1:
            raise ValueError("max_outer_steps must be >= 1")
        if self.all_fail_bonus < 0:
            raise ValueError("all_fail_bonus must be >= 0")
        if self.score_sampling_temperature <= 0:
            raise ValueError("score_sampling_temperature must be > 0")
        if self.num_parallel_branches < 1:
            raise ValueError("num_parallel_branches must be >= 1")
        if self.ucb_exploration_coef < 0:
            raise ValueError("ucb_exploration_coef must be >= 0")
        if self.train_rejection_max_retries < 1:
            raise ValueError("train_rejection_max_retries must be >= 1")
        if self.val_rejection_max_retries < 1:
            raise ValueError("val_rejection_max_retries must be >= 1")

    @classmethod
    def from_env(cls) -> "SampleWeightedGEPAConfig":
        return cls(
            candidate_pool_size=_env_int("AIME_SWG_CANDIDATE_POOL_SIZE", 5),
            reflect_train_batch_size=_env_int("AIME_SWG_REFLECT_TRAIN_BATCH_SIZE", 4),
            probe_val_batch_size=_env_int("AIME_SWG_PROBE_VAL_BATCH_SIZE", 4),
            inner_steps=_env_int("AIME_SWG_INNER_STEPS", 3),
            max_outer_steps=_env_int("AIME_SWG_MAX_OUTER_STEPS", 32),
            all_fail_bonus=_env_float("AIME_SWG_ALL_FAIL_BONUS", 0.1),
            score_sampling_temperature=_env_float("AIME_SWG_SCORE_SAMPLING_TEMPERATURE", 1.0),
            num_parallel_branches=_env_int("AIME_SWG_NUM_PARALLEL_BRANCHES", 4),
            ucb_exploration_coef=_env_float("AIME_SWG_UCB_EXPLORATION_COEF", 0.25),
            train_rejection_max_retries=_env_int("AIME_SWG_TRAIN_REJECTION_MAX_RETRIES", 8),
            val_rejection_max_retries=_env_int("AIME_SWG_VAL_REJECTION_MAX_RETRIES", 8),
        )
