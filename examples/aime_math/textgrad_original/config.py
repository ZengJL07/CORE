from __future__ import annotations

import os
from dataclasses import dataclass

from examples.aime_math.config import _env_bool, _env_int


@dataclass(frozen=True)
class OriginalTextGradConfig:
    algorithm: str
    batch_size: int
    max_epochs: int
    max_steps: int
    validation_frequency: int
    revert_on_validation_drop: bool

    def __post_init__(self) -> None:
        if self.algorithm != "tgd":
            raise ValueError(
                f"Original TextGrad path currently supports only AIME_TEXTGRAD_ALGORITHM='tgd', got {self.algorithm!r}"
            )
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.max_epochs < 1:
            raise ValueError(f"max_epochs must be >= 1, got {self.max_epochs}")
        if self.max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {self.max_steps}")
        if self.validation_frequency < 1:
            raise ValueError(f"validation_frequency must be >= 1, got {self.validation_frequency}")

    @classmethod
    def from_env(cls) -> "OriginalTextGradConfig":
        return cls(
            algorithm=os.environ.get("AIME_TEXTGRAD_ALGORITHM", "tgd").strip().lower(),
            batch_size=_env_int("AIME_TEXTGRAD_BATCH_SIZE", 3),
            max_epochs=_env_int("AIME_TEXTGRAD_MAX_EPOCHS", 3),
            max_steps=_env_int("AIME_TEXTGRAD_MAX_STEPS", 64),
            validation_frequency=_env_int("AIME_TEXTGRAD_VALIDATION_FREQUENCY", 1),
            revert_on_validation_drop=_env_bool("AIME_TEXTGRAD_REVERT_ON_VALIDATION_DROP", False),
        )
