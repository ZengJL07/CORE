from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_OUTPUT_BASE = Path("outputs/aime_math")
DEFAULT_DEEPSEEK_MODEL = "openai/deepseek-chat"
DEFAULT_DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"
DEFAULT_SHARED_CACHE_API_KEY = ""
DEFAULT_SOLVER_TEMPERATURE = 0.2
DEFAULT_REFLECTION_TEMPERATURE = 1.0
DEFAULT_MAX_TOKENS = 32000
DEFAULT_BASELINE_SCORE = 0.5666666666666667
DEFAULT_DATASET = "aime"
DEFAULT_MBPP_DATA_DIR = Path("/home/jlzeng/code/mbpptest/data")
DEFAULT_MBPP_HF_DATASET = "google-research-datasets/mbpp"
DEFAULT_MBPP_HF_CONFIG = "full"
DEFAULT_HMMT_FEB_2025_TEST_DATASET = "MathArena/hmmt_feb_2025"
DEFAULT_HMMT_FEB_2026_TEST_DATASET = "MathArena/hmmt_feb_2026"
DEFAULT_INITIAL_PROMPTS = {
    "aime": "Try to solve the math problem carefully. Break down the steps and provide the final answer as a single number.",
    "mbpp": (
        "Try to solve the Python programming task carefully. "
        "If a required function name is provided in the task, you must define exactly that function name. "
        "Provide the final answer as executable Python code with any required imports, "
        "wrapped in a ```python ... ``` block."
    ),
}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {value!r}") from exc


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


def _env_parallel_proposals(name: str, default: str | int) -> str | int:
    value = os.environ.get(name)
    if value is None:
        return default
    if value == "auto":
        return value
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer or 'auto', got {value!r}") from exc


def _env_best_candidate_strategy(name: str, default: str) -> str:
    value = os.environ.get(name, default).strip().lower()
    if value in {"default", "deepest"}:
        return value
    raise ValueError(f"{name} must be one of 'default' or 'deepest', got {value!r}")


def _env_optional_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer when set, got {value!r}") from exc


def _env_dataset(name: str, default: str) -> str:
    value = os.environ.get(name, default).strip().lower()
    if value in {"aime", "mbpp"}:
        return value
    raise ValueError(f"{name} must be one of 'aime' or 'mbpp', got {value!r}")


@dataclass(frozen=True)
class AIMEExperimentConfig:
    backend: str
    dataset_name: str
    enable_hmmt_feb_2025_test: bool
    enable_hmmt_feb_2026_test: bool
    mbpp_source: str
    mbpp_hf_dataset: str
    mbpp_hf_config: str | None
    mbpp_data_dir: Path
    max_train_examples: int | None
    max_val_examples: int | None
    max_test_examples: int | None
    seed: int
    max_metric_calls: int
    max_workers: int
    parallel_evaluation: bool
    solver_cache_enabled: bool
    eval_solver_cache_enabled: bool
    eval_pass_k: int
    skip_baseline_eval: bool
    default_baseline_pass_score: float
    default_baseline_mean_score: float
    best_candidate_strategy: str
    evaluate_existing_run_dir: Path | None
    evaluate_candidate_idx: int | None
    num_parallel_proposals: str | int
    solver_max_tokens: int
    solver_temperature: float
    reflection_temperature: float
    solver_model: str
    solver_api_base: str
    reflection_model: str
    output_root: Path
    run_id: str
    run_dir: Path
    solver_cache_dir: Path
    reflection_cache_dir: Path
    initial_prompt: str

    @classmethod
    def from_env(cls, backend: str) -> "AIMEExperimentConfig":
        dataset_name = _env_dataset("AIME_DATASET", DEFAULT_DATASET)
        output_base = Path(os.environ.get("AIME_OUTPUT_ROOT", str(DEFAULT_OUTPUT_BASE)))
        backend_root = output_base / dataset_name / backend
        seed = int(os.environ.get("AIME_SEED", "0"))
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        runs_base = os.environ.get("AIME_RUNS_DIR")
        run_dir = Path(runs_base) / run_id if runs_base else backend_root / "runs" / run_id
        initial_prompt = os.environ.get("AIME_INITIAL_PROMPT", DEFAULT_INITIAL_PROMPTS[dataset_name])
        return cls(
            backend=backend,
            dataset_name=dataset_name,
            enable_hmmt_feb_2025_test=_env_bool("AIME_ENABLE_HMMT_FEB_2025_TEST", False),
            enable_hmmt_feb_2026_test=_env_bool("AIME_ENABLE_HMMT_FEB_2026_TEST", False),
            mbpp_source=os.environ.get("AIME_MBPP_SOURCE", "huggingface").strip().lower(),
            mbpp_hf_dataset=os.environ.get("AIME_MBPP_HF_DATASET", DEFAULT_MBPP_HF_DATASET),
            mbpp_hf_config=os.environ.get("AIME_MBPP_HF_CONFIG", DEFAULT_MBPP_HF_CONFIG) or None,
            mbpp_data_dir=Path(os.environ.get("AIME_MBPP_DATA_DIR", str(DEFAULT_MBPP_DATA_DIR))),
            max_train_examples=_env_optional_int("AIME_MAX_TRAIN_EXAMPLES"),
            max_val_examples=_env_optional_int("AIME_MAX_VAL_EXAMPLES"),
            max_test_examples=_env_optional_int("AIME_MAX_TEST_EXAMPLES"),
            seed=seed,
            max_metric_calls=_env_int("AIME_MAX_METRIC_CALLS", 500),
            max_workers=_env_int("AIME_MAX_WORKERS", 16),
            parallel_evaluation=_env_bool("AIME_PARALLEL_EVALUATION", True),
            solver_cache_enabled=_env_bool("AIME_SOLVER_CACHE_ENABLED", False),
            eval_solver_cache_enabled=_env_bool("AIME_EVAL_SOLVER_CACHE_ENABLED", False),
            eval_pass_k=_env_int("AIME_EVAL_PASS_K", 3),
            skip_baseline_eval=_env_bool("AIME_SKIP_BASELINE_EVAL", False),
            default_baseline_pass_score=_env_float("AIME_DEFAULT_BASELINE_PASS_SCORE", DEFAULT_BASELINE_SCORE),
            default_baseline_mean_score=_env_float("AIME_DEFAULT_BASELINE_MEAN_SCORE", DEFAULT_BASELINE_SCORE),
            best_candidate_strategy=_env_best_candidate_strategy("AIME_BEST_CANDIDATE_STRATEGY", "default"),
            evaluate_existing_run_dir=(
                Path(os.environ["AIME_EVALUATE_EXISTING_RUN_DIR"])
                if os.environ.get("AIME_EVALUATE_EXISTING_RUN_DIR")
                else None
            ),
            evaluate_candidate_idx=_env_optional_int("AIME_EVALUATE_CANDIDATE_IDX"),
            num_parallel_proposals=_env_parallel_proposals("AIME_NUM_PARALLEL_PROPOSALS", "auto"),
            solver_max_tokens=_env_int("AIME_SOLVER_MAX_TOKENS", DEFAULT_MAX_TOKENS),
            solver_temperature=_env_float("AIME_SOLVER_TEMPERATURE", DEFAULT_SOLVER_TEMPERATURE),
            reflection_temperature=_env_float("AIME_REFLECTION_TEMPERATURE", DEFAULT_REFLECTION_TEMPERATURE),
            solver_model=os.environ.get("AIME_DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL),
            solver_api_base=os.environ.get("DEEPSEEK_API_BASE", DEFAULT_DEEPSEEK_API_BASE),
            reflection_model=os.environ.get(
                "AIME_REFLECTION_MODEL",
                os.environ.get("AIME_DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL),
            ),
            output_root=backend_root,
            run_id=run_id,
            run_dir=run_dir,
            solver_cache_dir=Path(os.environ.get("AIME_SOLVER_CACHE_DIR", str(backend_root / "api_cache" / f"seed_{seed}"))),
            reflection_cache_dir=Path(
                os.environ.get(
                    "AIME_REFLECTION_CACHE_DIR",
                    str(backend_root / "reflection_lm_cache" / f"seed_{seed}"),
                )
            ),
            initial_prompt=initial_prompt,
        )

    def solver_cache_namespace(self) -> dict[str, object]:
        return {
            "model": self.solver_model,
            "temperature": self.solver_temperature,
        }

    def solver_completion_kwargs(self, api_key: str) -> dict[str, object]:
        return {
            "api_key": api_key,
            "api_base": self.solver_api_base,
            "temperature": self.solver_temperature,
            "max_tokens": self.solver_max_tokens,
            "stream": True,
        }
