from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal


Trace2SkillMode = Literal["error", "success", "combined"]

DEFAULT_TRACE2SKILL_B = 6
DEFAULT_TRACE2SKILL_MODE: Trace2SkillMode = "combined"
DEFAULT_TRACE2SKILL_ERROR_ANALYST_MAX_TURNS = 3


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


@dataclass(frozen=True)
class Trace2SkillBaselineConfig:
    merge_fanin: int = DEFAULT_TRACE2SKILL_B
    mode: Trace2SkillMode = DEFAULT_TRACE2SKILL_MODE
    error_analyst_max_turns: int = DEFAULT_TRACE2SKILL_ERROR_ANALYST_MAX_TURNS

    @classmethod
    def from_env(cls) -> "Trace2SkillBaselineConfig":
        mode = os.environ.get("AIME_TRACE2SKILL_MODE", DEFAULT_TRACE2SKILL_MODE).strip().lower()
        if mode not in {"error", "success", "combined"}:
            raise ValueError(
                "AIME_TRACE2SKILL_MODE must be one of 'error', 'success', or 'combined', "
                f"got {mode!r}"
            )

        return cls(
            merge_fanin=max(2, _env_int("AIME_TRACE2SKILL_B", DEFAULT_TRACE2SKILL_B)),
            mode=mode,
            error_analyst_max_turns=max(
                2,
                _env_int(
                    "AIME_TRACE2SKILL_ERROR_ANALYST_MAX_TURNS",
                    DEFAULT_TRACE2SKILL_ERROR_ANALYST_MAX_TURNS,
                ),
            ),
        )
