import hashlib
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dspy
from dspy.utils.exceptions import AdapterParseError
import sympy as sp
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from datasets import load_dataset
from examples.aime_math.config import DEFAULT_DEEPSEEK_API_BASE, DEFAULT_SHARED_CACHE_API_KEY
from examples.aime_math.run_artifacts import (
    build_best_events,
    infer_valset_total_from_summary,
    load_sample_weighted_gepa_candidate_points,
)
from gepa.utils.code_execution import ExecutionMode, execute_code


DATA_DIR = Path(__file__).resolve().parent / "data"
TRAIN_JSONL = DATA_DIR / "aimo_validation_aime_train.jsonl"
TEST_JSONL = DATA_DIR / "aime_2025_test.jsonl"
HMMT_FEB_2025_TEST_JSONL = DATA_DIR / "hmmt_feb_2025_test.jsonl"
HMMT_FEB_2026_TEST_JSONL = DATA_DIR / "hmmt_feb_2026_test.jsonl"
API_CACHE_DIR = Path("outputs/aime_math/api_cache")
LEGACY_API_OUTPUT_DIR = Path("outputs/aime_math/api_outputs")
LEGACY_FITNESS_CACHE_JSON_DIR = Path("outputs/aime_math/fitness_cache_json")
_LEGACY_SOLVER_CACHE_ROOTS = (
    Path("outputs/aime_math/prompt_UCB/api_cache"),
    Path("outputs/aime_math/textgrad/api_cache"),
    Path("outputs/aime_math/textgrad_original/api_cache"),
    Path("outputs/aime_math/trace2skill_baseline/api_cache"),
)

_DEFAULT_SOLVER_CLIENT: "MathSolverClient | None" = None
_CACHE_VOLATILE_KEYS = frozenset({"api_base", "api_key"})
_DEFAULT_DATASET_TASK: "PromptOptimizationDatasetTask | None" = None
TEST_SOLVER_MODEL_PREFIX = "test://"
TEST_REFLECTION_MODEL_PREFIX = "test://"
HMMT_FEB_2025_DATASET = "MathArena/hmmt_feb_2025"
HMMT_FEB_2026_DATASET = "MathArena/hmmt_feb_2026"

# Default number of attempts for transient API failures (connection resets,
# 5xx, SSL EOF, rate limits, ...). Overridable per-client and via env vars in
# the launch scripts (AIME_SOLVER_API_MAX_RETRIES / AIME_REFLECTION_API_MAX_RETRIES).
DEFAULT_API_MAX_RETRIES = 5
# Exponential backoff bounds (seconds) used between transient-failure retries.
_API_RETRY_BASE_DELAY = 1.0
_API_RETRY_MAX_DELAY = 30.0


def _retry_delay_seconds(attempt: int) -> float:
    """Exponential backoff with a small deterministic jitter for retry ``attempt`` (1-indexed)."""
    delay = min(_API_RETRY_BASE_DELAY * (2 ** (attempt - 1)), _API_RETRY_MAX_DELAY)
    # Deterministic jitter (no global RNG dependency) keyed on the attempt index.
    jitter = (hash(("api_retry_jitter", attempt)) % 1000) / 1000.0 * 0.25 * delay
    return delay + jitter


def call_with_transient_retries(
    func,
    *,
    max_retries: int,
    description: str,
    log_prefix: str = "[AIME]",
):
    """Call ``func`` retrying only on transient (non-parse) exceptions.

    ``max_retries`` is the total number of attempts (>=1). ``AdapterParseError``
    is re-raised immediately so callers can run their own answer-recovery logic;
    every other exception is treated as transient and retried with exponential
    backoff until the attempts are exhausted, then re-raised.
    """
    attempts = max(1, int(max_retries))
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return func()
        except AdapterParseError:
            raise
        except Exception as exc:
            last_exc = exc
            first_line = str(exc).splitlines()[0][:200] if str(exc) else ""
            if attempt < attempts:
                delay = _retry_delay_seconds(attempt)
                print(
                    f"{log_prefix} Warning: {description} failed "
                    f"(attempt {attempt}/{attempts}): {type(exc).__name__}: {first_line}. "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
            else:
                print(
                    f"{log_prefix} Warning: {description} failed "
                    f"(attempt {attempt}/{attempts}): {type(exc).__name__}: {first_line}. "
                    "Retries exhausted."
                )
    assert last_exc is not None
    raise last_exc


class MathSolverSignature(dspy.Signature):
    input = dspy.InputField(desc="The math problem to solve.")
    answer = dspy.OutputField(desc="The final numerical answer.")


@dataclass(frozen=True)
class DatasetSplits:
    trainset: list[Any]
    valset: list[Any]
    testset: list[Any]


class PromptOptimizationDatasetTask:
    """Shared dataset/evaluation interface for all AIME prompt optimizers.

    This keeps dataset acquisition and scoring protocol explicit so GEPA,
    ParentReflectionGEPA (prompt_UCB), Trace2Skill, and TextGrad variants can
    all share the same backend and future tasks can swap in a different one.
    """

    def load_splits(self, *, seed: int = 0) -> DatasetSplits:
        raise NotImplementedError

    def evaluate_example(
        self,
        prompt: str,
        example: Any,
        *,
        use_solver_cache: bool | None = None,
        cache_extra: dict | None = None,
        lookup_cache_extra: dict | None = None,
    ) -> tuple[float, dict[str, Any]]:
        raise NotImplementedError

    def evaluate_dataset(
        self,
        prompt: str,
        dataset: list[Any],
        *,
        max_workers: int = 32,
        use_solver_cache: bool | None = None,
        pass_k: int = 1,
        return_stats: bool = False,
        cache_label: str | None = None,
    ):
        raise NotImplementedError

    def make_reflection_instruction(self, current_prompt: str, current_batch_feedback: str, current_batch_score: float) -> str:
        return current_prompt

    def trace2skill_trajectory_objective(self) -> str:
        return "improve task performance"

    def trace2skill_variable_role_description(self) -> str:
        return "system prompt to a language model"

    def trace2skill_success_filter_instruction(self) -> str:
        return (
            "This trajectory is SUCCESSFUL. Only return a suggestion if there is a concise, generalizable lesson "
            "that is likely to improve future trajectories. Otherwise return NONE."
        )

    def trace2skill_diagnosis_focus_instruction(self) -> str:
        return "We are interested in diagnosing why the current variable underperformed on this trajectory."

    def trace2skill_suggestion_instruction(self) -> str:
        return (
            "Turn the diagnosis into one concise, direct improvement suggestion for the prompt. "
            "The suggestion must be generalizable beyond this example. If no safe generalizable lesson exists, return NONE."
        )

    def trace2skill_merge_instruction(self) -> str:
        return (
            "Below is a small group of trajectory-local improvement suggestions for the prompt.\n"
            "Merge them into one concise, conflict-free, generalizable consolidated suggestion.\n"
            "Prefer recurring patterns. Drop advice that looks too instance-specific or redundant."
        )

    def trace2skill_rewrite_role_description(self) -> str:
        return "system prompt"

    def textgrad_evaluation_instruction(self) -> str:
        raise NotImplementedError

    def generate_model_output(
        self,
        example: Any,
        prompt: str,
        *,
        use_solver_cache: bool | None = None,
        cache_extra: dict | None = None,
        lookup_cache_extra: dict | None = None,
    ) -> tuple[str, str]:
        raise NotImplementedError

    def test_stub_answer(self, example: Any) -> str:
        return "0"

    def test_stub_failure_answer(self, example: Any) -> str:
        return "1"


def _build_solver_instructions(prompt: str, include_reasoning: bool, *, output_mode: str = "integer") -> str:
    prompt = prompt.strip()
    if output_mode == "python_code":
        if include_reasoning:
            schema_hint = '{"reasoning": "...", "answer": "```python\\n<code>\\n```"}'
            fields_hint = "reasoning, answer"
        else:
            schema_hint = '{"answer": "```python\\n<code>\\n```"}'
            fields_hint = "answer"
        return (
            f"{prompt}\n\n"
            "Output format requirements (must follow exactly):\n"
            "- Return ONLY a valid JSON object.\n"
            f"- Include exactly these keys: {fields_hint}.\n"
            "- `answer` must contain exactly one fenced Python code block using ```python ... ```.\n"
            "- Do not put explanatory text outside the JSON object.\n"
            f"- Example format: {schema_hint}\n"
        )

    if include_reasoning:
        schema_hint = '{"reasoning": "...", "answer": "..."}'
        fields_hint = "reasoning, answer"
    else:
        schema_hint = '{"answer": "..."}'
        fields_hint = "answer"

    return (
        f"{prompt}\n\n"
        "Output format requirements (must follow exactly):\n"
        "- Return ONLY a valid JSON object.\n"
        f"- Include exactly these keys: {fields_hint}.\n"
        "- `answer` must contain only the final exact answer, with no extra prose.\n"
        "- Use a plain integer when the problem asks for an integer answer; otherwise use a single exact mathematical expression.\n"
        "- Allowed exact forms include integers, fractions, radicals, `pi`, factorials, or comma-separated exact values when the problem has multiple answers.\n"
        "- Do not output markdown or a full solution inside `answer`.\n"
        f"- Example format: {schema_hint}\n"
    )


def _prediction_cache_key(
    problem_input: str,
    prompt: str,
    namespace: dict | None = None,
    cache_extra: dict | None = None,
    output_mode: str = "integer",
) -> str:
    payload = {"input": problem_input, "prompt": prompt, "output_mode": output_mode}
    if namespace:
        payload["namespace"] = namespace
    if cache_extra:
        payload["cache_extra"] = cache_extra
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _normalized_cache_mapping(mapping: dict | None) -> dict:
    if not mapping:
        return {}
    return {k: v for k, v in mapping.items() if k not in _CACHE_VOLATILE_KEYS}


def _backend_agnostic_cache_mapping(mapping: dict | None) -> dict:
    normalized = _normalized_cache_mapping(mapping)
    return {k: v for k, v in normalized.items() if k != "backend"}


def _prediction_from_payload(payload: dict, *, output_mode: str = "integer") -> dspy.Prediction:
    answer, reasoning = _normalize_prediction_fields(
        str(payload.get("answer", "")),
        str(payload.get("reasoning", "")),
        output_mode=output_mode,
    )
    return dspy.Prediction(
        answer=answer,
        reasoning=reasoning,
    )


def _reextract_cached_prediction(payload: dict, *, output_mode: str = "integer") -> dspy.Prediction:
    prediction = _prediction_from_payload(payload, output_mode=output_mode)
    answer = str(getattr(prediction, "answer", "")).strip()
    if answer and not answer.startswith('": "'):
        return prediction

    raw_candidates = [
        str(payload.get("answer", "")).strip(),
        str(payload.get("reasoning", "")).strip(),
    ]
    for candidate in raw_candidates:
        reparsed = _prediction_from_raw_lm_response(candidate, output_mode=output_mode)
        if reparsed is None:
            continue
        reparsed_answer = str(getattr(reparsed, "answer", "")).strip()
        if reparsed_answer:
            return reparsed
    return prediction


def _normalize_prediction_fields(answer: str, reasoning: str = "", *, output_mode: str = "integer") -> tuple[str, str]:
    """Normalize nested structured answers into the plain integer answer field."""
    answer = _repair_partial_answer_fragment(str(answer).strip())
    reasoning = str(reasoning)

    for _ in range(2):
        try:
            payload = json.loads(answer)
        except json.JSONDecodeError:
            break

        if isinstance(payload, dict) and "answer" in payload:
            nested_reasoning = payload.get("reasoning")
            if nested_reasoning and not reasoning:
                reasoning = str(nested_reasoning)
            answer = str(payload.get("answer", "")).strip()
            continue

        if isinstance(payload, int):
            answer = str(payload)
        break

    if output_mode == "python_code":
        fenced = re.search(r"```python\s*(.*?)```", answer, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            return fenced.group(1).strip(), reasoning
        generic_fence = re.search(r"```\s*(.*?)```", answer, flags=re.DOTALL)
        if generic_fence:
            return generic_fence.group(1).strip(), reasoning
        return answer.strip(), reasoning

    boxed_expr = _extract_boxed_expression(answer)
    if boxed_expr is not None:
        return boxed_expr, reasoning or answer[:1000]

    if not re.fullmatch(r"-?\d+", answer):
        answer_match = re.search(r"(?is)(?:final answer|answer)\s*[:=]?\s*(.+)$", answer)
        if answer_match:
            return answer_match.group(1).strip(), reasoning or answer[:1000]

    return answer, reasoning


def _repair_partial_answer_fragment(answer: str) -> str:
    if not answer:
        return answer

    stripped = answer.strip()
    partial_json_match = re.search(r'"\s*:\s*"(?P<value>.*?)"\s*}?\s*(?:```)?\s*$', stripped, flags=re.DOTALL)
    if partial_json_match:
        return partial_json_match.group("value").strip()

    partial_json_open_match = re.search(r'"\s*:\s*"(?P<value>.+)$', stripped, flags=re.DOTALL)
    if partial_json_open_match:
        return partial_json_open_match.group("value").strip()

    answer_key_match = re.search(r'"answer"\s*:\s*"(?P<value>.*?)"', stripped, flags=re.DOTALL)
    if answer_key_match:
        return answer_key_match.group("value").strip()

    return stripped


def _raw_lm_response_from_parse_error(exc: AdapterParseError) -> str:
    text = str(exc)
    marker = "LM Response:"
    if marker not in text:
        return ""

    raw = text.split(marker, 1)[1]
    for stop_marker in ("\n\nExpected to find output fields", "\n\nExpected"):
        if stop_marker in raw:
            raw = raw.split(stop_marker, 1)[0]
            break
    return raw.strip()


def _prediction_from_raw_lm_response(raw_response: str, *, output_mode: str = "integer") -> dspy.Prediction | None:
    if not raw_response:
        return None

    raw_response = raw_response.strip()
    fence_match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", raw_response, flags=re.DOTALL)
    if fence_match:
        raw_response = fence_match.group(1).strip()

    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError:
        payload = None

    if isinstance(payload, dict):
        if "answer" in payload:
            answer, reasoning = _normalize_prediction_fields(
                str(payload.get("answer", "")),
                str(payload.get("reasoning", "")),
                output_mode=output_mode,
            )
            return dspy.Prediction(
                answer=answer,
                reasoning=reasoning,
            )
        return None

    if output_mode == "python_code":
        fenced = re.search(r"```python\s*(.*?)```", raw_response, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            return dspy.Prediction(answer=fenced.group(1).strip(), reasoning="")
        generic_fence = re.search(r"```\s*(.*?)```", raw_response, flags=re.DOTALL)
        if generic_fence:
            return dspy.Prediction(answer=generic_fence.group(1).strip(), reasoning="")
        return dspy.Prediction(answer=raw_response.strip(), reasoning="")

    if isinstance(payload, int):
        return dspy.Prediction(answer=str(payload), reasoning="")

    stripped = raw_response.strip()
    answer, reasoning = _normalize_prediction_fields(stripped, "", output_mode=output_mode)
    if answer:
        return dspy.Prediction(answer=answer, reasoning=reasoning)
    return None


def _iter_legacy_cache_payloads():
    if LEGACY_API_OUTPUT_DIR.exists():
        for path in sorted(LEGACY_API_OUTPUT_DIR.glob("*.json")):
            try:
                with path.open("r", encoding="utf-8") as f:
                    payload = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue

            yield payload, path

    if LEGACY_FITNESS_CACHE_JSON_DIR.exists():
        for path in sorted(LEGACY_FITNESS_CACHE_JSON_DIR.glob("*.json")):
            try:
                with path.open("r", encoding="utf-8") as f:
                    payload = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue

            result = payload.get("data", {}).get("result", [])
            if not isinstance(result, list) or len(result) < 3 or not isinstance(result[2], dict):
                continue

            side_info = result[2]
            yield {
                "input": side_info.get("input", ""),
                "prompt": side_info.get("prompt", ""),
                "answer": side_info.get("output", ""),
                "reasoning": side_info.get("reasoning", ""),
            }, path


class MathSolverClient:
    """Thin API client wrapper that owns request-level response caching."""

    def __init__(
        self,
        cache_dir: Path = API_CACHE_DIR,
        cache_namespace: dict | None = None,
        model_name: str | None = None,
        completion_kwargs: dict | None = None,
        enable_cache: bool = True,
        output_mode: str = "integer",
        task: PromptOptimizationDatasetTask | None = None,
        api_max_retries: int = DEFAULT_API_MAX_RETRIES,
    ):
        self.cache_dir = cache_dir
        self.raw_cache_namespace = dict(cache_namespace or {})
        self.cache_namespace = _normalized_cache_mapping(self.raw_cache_namespace)
        self.model_name = model_name
        self.completion_kwargs = dict(completion_kwargs or {})
        self.enable_cache = enable_cache
        self.output_mode = output_mode
        self.task = task
        self.api_max_retries = max(1, int(api_max_retries))
        self._lock = threading.RLock()
        self._phase_agnostic_index: dict[str, list[Path]] | None = None
        self._cross_root_phase_agnostic_index: dict[str, list[Path]] | None = None
        self._cross_root_exact_index: dict[str, list[Path]] | None = None

    def migrate_legacy_caches(self) -> tuple[int, int]:
        migrated = 0
        skipped = 0
        for payload, source_path in _iter_legacy_cache_payloads():
            problem_input = str(payload.get("input", ""))
            prompt = str(payload.get("prompt", ""))
            if not problem_input or not prompt:
                skipped += 1
                continue

            prediction = _prediction_from_payload(payload, output_mode=self.output_mode)
            if not getattr(prediction, "answer", "") and not getattr(prediction, "reasoning", ""):
                skipped += 1
                continue

            cache_path = self._cache_path(problem_input, prompt)
            if cache_path.exists():
                skipped += 1
                continue

            self._save_prediction_cache(
                problem_input,
                prompt,
                prediction,
                source=f"legacy:{source_path.parent.name}/{source_path.name}",
            )
            migrated += 1

        return migrated, skipped

    def predict(
        self,
        example,
        prompt: str,
        max_retries: int = 1,
        use_cache: bool | None = None,
        cache_extra: dict | None = None,
        lookup_cache_extra: dict | None = None,
    ):
        problem_input = str(example.input)
        cache_enabled = self.enable_cache if use_cache is None else use_cache
        if cache_enabled:
            cached_prediction = self._load_prediction_cache(
                problem_input,
                prompt,
                cache_extra=cache_extra,
                lookup_cache_extra=lookup_cache_extra,
            )
            if cached_prediction is not None:
                print("[AIME] Reusing cached API prediction.")
                return cached_prediction

        answer_instructions = _build_solver_instructions(prompt, include_reasoning=True, output_mode=self.output_mode)
        cache_empty_failure = True
        for attempt in range(1, max_retries + 1):
            try:
                prediction = call_with_transient_retries(
                    lambda: self._call_solver(example, answer_instructions, include_reasoning=True),
                    max_retries=self.api_max_retries,
                    description="solver API call",
                )
                normalized_prediction = self._normalize_prediction(prediction)
                if cache_enabled:
                    self._save_prediction_cache(
                        problem_input,
                        prompt,
                        normalized_prediction,
                        source="live_api",
                        cache_extra=cache_extra,
                    )
                return normalized_prediction
            except AdapterParseError as exc:
                raw_response = _raw_lm_response_from_parse_error(exc)
                recovered_prediction = _prediction_from_raw_lm_response(raw_response, output_mode=self.output_mode)
                if recovered_prediction is not None:
                    print("[AIME] Recovered answer from non-JSON LM response.")
                    if cache_enabled:
                        self._save_prediction_cache(
                            problem_input,
                            prompt,
                            recovered_prediction,
                            source="live_api_recovered_parse_error",
                            cache_extra=cache_extra,
                        )
                    return recovered_prediction
                print(
                    "[AIME] Warning: answer parse failure "
                    f"(attempt {attempt}/{max_retries}); raw response: {raw_response[:200]!r}"
                )
            except Exception as exc:
                # Transient API failures were already retried up to api_max_retries
                # times inside call_with_transient_retries; reaching here means they
                # were exhausted. Do not poison the cache with an empty answer.
                cache_empty_failure = False
                print(
                    "[AIME] Warning: solver API call failed after "
                    f"{self.api_max_retries} attempt(s) "
                    f"(parse-attempt {attempt}/{max_retries}): "
                    f"{type(exc).__name__}: {str(exc).splitlines()[0][:200] if str(exc) else ''}"
                )

        print("[AIME] Warning: exhausted parse retries; returning empty prediction.")
        empty_prediction = dspy.Prediction(answer="", reasoning="")
        if cache_enabled and cache_empty_failure:
            self._save_prediction_cache(
                problem_input,
                prompt,
                empty_prediction,
                source="parse_failure_empty",
                cache_extra=cache_extra,
            )
        return empty_prediction

    def _cache_path(
        self,
        problem_input: str,
        prompt: str,
        namespace: dict | None = None,
        cache_extra: dict | None = None,
    ) -> Path:
        effective_namespace = self.cache_namespace if namespace is None else namespace
        return self.cache_dir / f"{_prediction_cache_key(problem_input, prompt, effective_namespace, cache_extra, self.output_mode)}.json"

    def _legacy_cache_paths(
        self,
        problem_input: str,
        prompt: str,
        cache_extra: dict | None = None,
    ) -> list[Path]:
        namespaces: list[dict] = []
        if self.raw_cache_namespace:
            namespaces.append(dict(self.raw_cache_namespace))

        current_api_base = self.completion_kwargs.get("api_base") or os.environ.get("DEEPSEEK_API_BASE")
        for candidate_api_base in (DEFAULT_DEEPSEEK_API_BASE, current_api_base):
            if not candidate_api_base:
                continue
            legacy_namespace = dict(self.cache_namespace)
            legacy_namespace["api_base"] = candidate_api_base
            if legacy_namespace not in namespaces:
                namespaces.append(legacy_namespace)

        paths = []
        for namespace in namespaces:
            path = self._cache_path(problem_input, prompt, namespace=namespace, cache_extra=cache_extra)
            if path not in paths:
                paths.append(path)
        return paths

    def _cross_root_exact_paths(
        self,
        problem_input: str,
        prompt: str,
        cache_extra: dict | None = None,
    ) -> list[Path]:
        index = self._get_cross_root_exact_index()
        paths: list[Path] = []
        for namespace in self._lookup_namespace_variants(self.cache_namespace):
            key = _prediction_cache_key(problem_input, prompt, namespace, cache_extra, self.output_mode)
            for path in index.get(key, []):
                if path not in paths:
                    paths.append(path)
        return paths

    def _phase_agnostic_legacy_paths(
        self,
        problem_input: str,
        prompt: str,
        lookup_cache_extra: dict | None,
    ) -> list[Path]:
        if not lookup_cache_extra:
            return []
        if "pass_k" not in lookup_cache_extra or "attempt_idx" not in lookup_cache_extra:
            return []

        index = self._get_phase_agnostic_index()
        paths: list[Path] = []
        for namespace in self._lookup_namespace_variants(self.cache_namespace):
            key = _prediction_cache_key(problem_input, prompt, namespace, lookup_cache_extra, self.output_mode)
            for path in index.get(key, []):
                if path not in paths:
                    paths.append(path)
            for path in self._get_cross_root_phase_agnostic_index().get(key, []):
                if path not in paths:
                    paths.append(path)
        return paths

    def _get_phase_agnostic_index(self) -> dict[str, list[Path]]:
        cached = self._phase_agnostic_index
        if cached is not None:
            return cached

        index: dict[str, list[Path]] = {}
        if self.cache_dir.exists():
            for path in self.cache_dir.glob("*.json"):
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue

                cache_extra = payload.get("cache_extra")
                if not isinstance(cache_extra, dict):
                    continue
                if "phase" not in cache_extra or "pass_k" not in cache_extra or "attempt_idx" not in cache_extra:
                    continue

                lookup_cache_extra = {
                    "pass_k": cache_extra["pass_k"],
                    "attempt_idx": cache_extra["attempt_idx"],
                }
                namespace = payload.get("namespace")
                normalized_namespace = _normalized_cache_mapping(namespace) if isinstance(namespace, dict) else None
                key = _prediction_cache_key(
                    str(payload.get("input", "")),
                    str(payload.get("prompt", "")),
                    normalized_namespace,
                    lookup_cache_extra,
                    str(payload.get("output_mode", self.output_mode)),
                )
                index.setdefault(key, []).append(path)

        self._phase_agnostic_index = index
        return index

    def _get_cross_root_phase_agnostic_index(self) -> dict[str, list[Path]]:
        cached = self._cross_root_phase_agnostic_index
        if cached is not None:
            return cached

        current_cache_dir = self.cache_dir.resolve()
        current_seed_dir_name = self.cache_dir.name
        index: dict[str, list[Path]] = {}
        for root in _LEGACY_SOLVER_CACHE_ROOTS:
            candidate_dir = root / current_seed_dir_name
            try:
                resolved_dir = candidate_dir.resolve()
            except OSError:
                continue
            if resolved_dir == current_cache_dir or not candidate_dir.exists():
                continue

            for path in candidate_dir.glob("*.json"):
                self._add_phase_agnostic_index_entry(index, path)

        self._cross_root_phase_agnostic_index = index
        return index

    def _get_cross_root_exact_index(self) -> dict[str, list[Path]]:
        cached = self._cross_root_exact_index
        if cached is not None:
            return cached

        current_cache_dir = self.cache_dir.resolve()
        current_seed_dir_name = self.cache_dir.name
        index: dict[str, list[Path]] = {}
        candidate_dirs = [self.cache_dir]
        for root in _LEGACY_SOLVER_CACHE_ROOTS:
            candidate_dir = root / current_seed_dir_name
            try:
                resolved_dir = candidate_dir.resolve()
            except OSError:
                continue
            if resolved_dir == current_cache_dir or not candidate_dir.exists():
                continue
            candidate_dirs.append(candidate_dir)

        for candidate_dir in candidate_dirs:
            if not candidate_dir.exists():
                continue
            for path in candidate_dir.glob("*.json"):
                self._add_exact_index_entry(index, path)

        self._cross_root_exact_index = index
        return index

    @staticmethod
    def _lookup_namespace_variants(namespace: dict | None) -> list[dict | None]:
        variants: list[dict | None] = []
        exact = _normalized_cache_mapping(namespace)
        backend_agnostic = _backend_agnostic_cache_mapping(namespace)
        for item in (exact, backend_agnostic):
            normalized_item = item or None
            if normalized_item not in variants:
                variants.append(normalized_item)
        return variants

    @staticmethod
    def _add_phase_agnostic_index_entry(index: dict[str, list[Path]], path: Path) -> None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return

        cache_extra = payload.get("cache_extra")
        if not isinstance(cache_extra, dict):
            return
        if "phase" not in cache_extra or "pass_k" not in cache_extra or "attempt_idx" not in cache_extra:
            return

        lookup_cache_extra = {
            "pass_k": cache_extra["pass_k"],
            "attempt_idx": cache_extra["attempt_idx"],
        }
        namespace = payload.get("namespace")
        for namespace_variant in MathSolverClient._lookup_namespace_variants(namespace if isinstance(namespace, dict) else None):
            key = _prediction_cache_key(
                str(payload.get("input", "")),
                str(payload.get("prompt", "")),
                namespace_variant,
                lookup_cache_extra,
                str(payload.get("output_mode", "integer")),
            )
            index.setdefault(key, []).append(path)

    @staticmethod
    def _add_exact_index_entry(index: dict[str, list[Path]], path: Path) -> None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return

        namespace = payload.get("namespace")
        cache_extra = payload.get("cache_extra")
        if cache_extra is not None and not isinstance(cache_extra, dict):
            return

        for namespace_variant in MathSolverClient._lookup_namespace_variants(namespace if isinstance(namespace, dict) else None):
            key = _prediction_cache_key(
                str(payload.get("input", "")),
                str(payload.get("prompt", "")),
                namespace_variant,
                cache_extra,
                str(payload.get("output_mode", "integer")),
            )
            index.setdefault(key, []).append(path)

    def _load_prediction_cache(
        self,
        problem_input: str,
        prompt: str,
        cache_extra: dict | None = None,
        lookup_cache_extra: dict | None = None,
    ) -> dspy.Prediction | None:
        def try_paths(candidate_paths: list[Path]) -> dspy.Prediction | None:
            seen: set[Path] = set()
            for cache_path in candidate_paths:
                if cache_path in seen:
                    continue
                seen.add(cache_path)
                if not cache_path.exists():
                    continue

                try:
                    with cache_path.open("r", encoding="utf-8") as f:
                        payload = json.load(f)
                    payload_output_mode = str(payload.get("output_mode", self.output_mode))
                    return _reextract_cached_prediction(payload, output_mode=payload_output_mode)
                except (OSError, json.JSONDecodeError):
                    continue
            return None

        direct_paths = [
            self._cache_path(problem_input, prompt, cache_extra=cache_extra),
            *self._legacy_cache_paths(problem_input, prompt, cache_extra=cache_extra),
        ]
        cached_prediction = try_paths(direct_paths)
        if cached_prediction is not None:
            return cached_prediction

        fallback_paths = [
            *self._cross_root_exact_paths(problem_input, prompt, cache_extra=cache_extra),
            *self._phase_agnostic_legacy_paths(problem_input, prompt, lookup_cache_extra),
        ]
        return try_paths(fallback_paths)

    def _save_prediction_cache(
        self,
        problem_input: str,
        prompt: str,
        prediction,
        source: str,
        cache_extra: dict | None = None,
    ) -> Path:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self._cache_path(problem_input, prompt, cache_extra=cache_extra)
        payload = {
            "input": problem_input,
            "prompt": prompt,
            "output_mode": self.output_mode,
            "answer": getattr(prediction, "answer", ""),
            "reasoning": getattr(prediction, "reasoning", ""),
            "source": source,
        }
        if self.cache_namespace:
            payload["namespace"] = self.cache_namespace
        if cache_extra:
            payload["cache_extra"] = cache_extra
        tmp_path = cache_path.with_suffix(f".{threading.get_ident()}.tmp")
        with self._lock:
            with tmp_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            tmp_path.replace(cache_path)
            self._phase_agnostic_index = None
            self._cross_root_phase_agnostic_index = None
            self._cross_root_exact_index = None
        return cache_path

    def _normalize_prediction(self, prediction) -> dspy.Prediction:
        answer, reasoning = _normalize_prediction_fields(
            str(getattr(prediction, "answer", "")),
            str(getattr(prediction, "reasoning", "")),
            output_mode=self.output_mode,
        )
        return dspy.Prediction(answer=answer, reasoning=reasoning)

    def _call_solver(self, example, instructions: str, include_reasoning: bool):
        if self.model_name is not None and self.model_name.startswith(TEST_SOLVER_MODEL_PREFIX):
            return _build_test_solver_response(
                model_name=self.model_name,
                example=example,
                prompt=instructions,
                output_mode=self.output_mode,
                task=self.task,
            )
        if self.model_name is not None:
            return self._call_litellm_solver(example, instructions)

        signature = MathSolverSignature.with_instructions(instructions)
        predictor = dspy.ChainOfThought(signature) if include_reasoning else dspy.Predict(signature)
        return predictor(input=example.input)

    def _call_litellm_solver(self, example, instructions: str):
        import litellm

        response = litellm.completion(
            model=self.model_name,
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": str(example.input)},
            ],
            **self.completion_kwargs,
        )
        if self.completion_kwargs.get("stream"):
            content_parts = []
            reasoning_parts = []
            for chunk in response:
                delta = chunk.choices[0].delta
                content = getattr(delta, "content", None)
                reasoning = getattr(delta, "reasoning_content", None)
                if content:
                    content_parts.append(content)
                if reasoning:
                    reasoning_parts.append(reasoning)

            raw_response = "".join(content_parts)
            hidden_reasoning = "".join(reasoning_parts)
            prediction = _prediction_from_raw_lm_response(raw_response, output_mode=self.output_mode)
            if prediction is not None:
                reasoning = getattr(prediction, "reasoning", "")
                if hidden_reasoning:
                    reasoning = f"{hidden_reasoning}\n\n{reasoning}".strip()
                return dspy.Prediction(answer=prediction.answer, reasoning=reasoning)
            return dspy.Prediction(answer="", reasoning=(hidden_reasoning or raw_response)[:1000])

        raw_response = response.choices[0].message.content or ""
        prediction = _prediction_from_raw_lm_response(raw_response, output_mode=self.output_mode)
        if prediction is not None:
            return prediction
        return dspy.Prediction(answer="", reasoning=raw_response[:1000])


class CachedLanguageModel:
    """Request-level cache wrapper for GEPA proposal/reflection LM calls."""

    def __init__(self, model_name: str, cache_dir: Path, *, api_max_retries: int = DEFAULT_API_MAX_RETRIES, **lm_kwargs):
        from gepa.optimize_anything import make_litellm_lm

        self.model_name = model_name
        self.raw_lm_kwargs = dict(lm_kwargs)
        self.lm_kwargs = _normalized_cache_mapping(self.raw_lm_kwargs)
        self.cache_dir = cache_dir
        self.api_max_retries = max(1, int(api_max_retries))
        # Disable the underlying LM's own retry loop so we don't compound it with
        # ours (api_max_retries attempts here). num_retries is an LM constructor
        # arg, not a completion kwarg, so it stays out of the request cache key.
        self._lm = make_litellm_lm(model_name, num_retries=0, **lm_kwargs)
        self._lock = threading.RLock()

    @property
    def total_cost(self) -> float:
        return float(getattr(self._lm, "total_cost", 0.0))

    def __call__(self, prompt) -> str:
        cache_path = self._cache_path(prompt)
        cached_response = self._load(prompt, cache_path)
        if cached_response is not None:
            print("[AIME] Reusing cached reflection LM response.")
            return cached_response

        if self.model_name.startswith(TEST_REFLECTION_MODEL_PREFIX):
            response = _build_test_reflection_response(model_name=self.model_name, prompt=str(prompt))
            self._save(cache_path, prompt, response)
            return response

        response = call_with_transient_retries(
            lambda: self._lm(prompt),
            max_retries=self.api_max_retries,
            description="reflection LM call",
        )
        self._save(cache_path, prompt, response)
        return response

    def _cache_path(self, prompt, lm_kwargs: dict | None = None) -> Path:
        effective_lm_kwargs = self.lm_kwargs if lm_kwargs is None else lm_kwargs
        payload = {
            "model": self.model_name,
            "lm_kwargs": effective_lm_kwargs,
            "prompt": prompt,
        }
        blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
        return self.cache_dir / f"{hashlib.sha256(blob).hexdigest()}.json"

    def _legacy_cache_paths(self, prompt) -> list[Path]:
        candidate_kwargs: list[dict] = []
        if self.raw_lm_kwargs:
            candidate_kwargs.append(dict(self.raw_lm_kwargs))

        current_api_base = self.raw_lm_kwargs.get("api_base") or os.environ.get("DEEPSEEK_API_BASE")
        current_api_key = self.raw_lm_kwargs.get("api_key") or os.environ.get("DEEPSEEK_API_KEY")
        candidate_api_bases = [value for value in (DEFAULT_DEEPSEEK_API_BASE, current_api_base) if value]
        candidate_api_keys = [value for value in (DEFAULT_SHARED_CACHE_API_KEY, current_api_key) if value]

        for candidate_api_base in candidate_api_bases or [None]:
            for candidate_api_key in candidate_api_keys or [None]:
                legacy_kwargs = dict(self.lm_kwargs)
                if candidate_api_base:
                    legacy_kwargs["api_base"] = candidate_api_base
                if candidate_api_key:
                    legacy_kwargs["api_key"] = candidate_api_key
                if legacy_kwargs not in candidate_kwargs:
                    candidate_kwargs.append(legacy_kwargs)

        paths = []
        for kwargs in candidate_kwargs:
            path = self._cache_path(prompt, lm_kwargs=kwargs)
            if path not in paths:
                paths.append(path)
        return paths

    def _load(self, prompt, cache_path: Path) -> str | None:
        for candidate_path in [cache_path, *self._legacy_cache_paths(prompt)]:
            if not candidate_path.exists():
                continue
            try:
                with candidate_path.open("r", encoding="utf-8") as f:
                    payload = json.load(f)
                response = payload.get("response")
                if isinstance(response, str):
                    return response
            except (OSError, json.JSONDecodeError):
                continue
        return None

    def _save(self, cache_path: Path, prompt, response: str) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model_name,
            "lm_kwargs": self.lm_kwargs,
            "prompt": prompt,
            "response": response,
        }
        tmp_path = cache_path.with_suffix(f".{threading.get_ident()}.tmp")
        with self._lock:
            with tmp_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
            tmp_path.replace(cache_path)


def configure_default_solver_client(
    cache_dir: Path = API_CACHE_DIR,
    cache_namespace: dict | None = None,
    model_name: str | None = None,
    completion_kwargs: dict | None = None,
    enable_cache: bool = True,
    output_mode: str = "integer",
    task: PromptOptimizationDatasetTask | None = None,
    api_max_retries: int = DEFAULT_API_MAX_RETRIES,
) -> None:
    global _DEFAULT_SOLVER_CLIENT
    _DEFAULT_SOLVER_CLIENT = MathSolverClient(
        cache_dir=cache_dir,
        cache_namespace=cache_namespace,
        model_name=model_name,
        completion_kwargs=completion_kwargs,
        enable_cache=enable_cache,
        output_mode=output_mode,
        task=task,
        api_max_retries=api_max_retries,
    )


def get_default_solver_client() -> MathSolverClient:
    global _DEFAULT_SOLVER_CLIENT
    if _DEFAULT_SOLVER_CLIENT is None:
        _DEFAULT_SOLVER_CLIENT = MathSolverClient()
    return _DEFAULT_SOLVER_CLIENT


def migrate_existing_api_caches() -> tuple[int, int]:
    """Move legacy JSON cache snapshots into the request-level API cache."""
    return get_default_solver_client().migrate_legacy_caches()


def run_llm(
    example,
    prompt: str,
    max_retries: int = 1,
    use_solver_cache: bool | None = None,
    cache_extra: dict | None = None,
    lookup_cache_extra: dict | None = None,
):
    """Run the LLM on a single example with the given prompt."""
    return get_default_solver_client().predict(
        example,
        prompt,
        max_retries=max_retries,
        use_cache=use_solver_cache,
        cache_extra=cache_extra,
        lookup_cache_extra=lookup_cache_extra,
    )


def math_metric(example, prediction):
    """Compute score and detailed feedback for math problems."""
    correct_answer = str(getattr(example, "answer", "")).strip()
    written_solution = getattr(example, "solution", "")
    solution_suffix = (
        f" Here's the full step-by-step solution:\n{written_solution}\n\nThink about what takeaways you can learn from this solution to improve your future answers and approach to similar problems"
        if written_solution
        else ""
    )
    prediction_answer = str(getattr(prediction, "answer", "")).strip()

    exact_match, expects_integer = _math_answers_match(correct_answer, prediction_answer)
    if exact_match is None:
        if expects_integer:
            feedback_text = (
                f"The final answer must be a valid integer and nothing else. You responded with '{prediction_answer}', "
                f"which couldn't be parsed as a python integer. Please ensure your answer is a valid integer without "
                f"any additional text or formatting. The correct answer is '{correct_answer}'."
                f"{solution_suffix}{' and ensure your final answer is a valid integer.' if written_solution else ''}"
            )
            return 0.0, feedback_text
        feedback_text = (
            f"The final answer must be a valid mathematical expression matching the expected value. You responded with "
            f"'{prediction_answer}', which couldn't be parsed reliably. Please provide only the final exact answer, "
            f"using standard mathematical notation such as integers, fractions, radicals, factorials, or comma-separated "
            f"exact values when the problem has multiple answers. The correct answer is '{correct_answer}'.{solution_suffix}"
        )
        return 0.0, feedback_text

    score = float(exact_match)
    status = "correct" if score == 1.0 else "incorrect"
    feedback_text = f"Your answer is {status}. The correct answer is '{correct_answer}'.{solution_suffix}"
    return score, feedback_text


def _math_answers_match(correct_answer: str, prediction_answer: str) -> tuple[bool | None, bool]:
    expects_integer = _looks_like_integer_answer(correct_answer)

    if expects_integer:
        try:
            return int(correct_answer) == int(prediction_answer), True
        except (ValueError, TypeError):
            return None, True

    expected_values = _parse_math_answer_set(correct_answer)
    predicted_values = _parse_math_answer_set(prediction_answer)
    if expected_values is None or predicted_values is None:
        return None, False
    if len(expected_values) != len(predicted_values):
        return False, False
    return _multiset_expr_equal(expected_values, predicted_values), False


def _looks_like_integer_answer(answer: str) -> bool:
    return bool(re.fullmatch(r"-?\d+", str(answer).strip()))


def _multiset_expr_equal(left: list[sp.Expr], right: list[sp.Expr]) -> bool:
    used = [False] * len(right)
    for left_expr in left:
        matched = False
        for idx, right_expr in enumerate(right):
            if used[idx]:
                continue
            if _expr_equal(left_expr, right_expr):
                used[idx] = True
                matched = True
                break
        if not matched:
            return False
    return True


def _expr_equal(left: sp.Expr, right: sp.Expr) -> bool:
    try:
        return bool(sp.simplify(left - right) == 0)
    except Exception:
        return False


def _parse_math_answer_set(answer: str) -> list[sp.Expr] | None:
    normalized = _normalize_math_answer_text(answer)
    if not normalized:
        return None
    parts = [part.strip() for part in normalized.split(",") if part.strip()]
    if not parts:
        return None

    expressions: list[sp.Expr] = []
    for part in parts:
        expr = _parse_math_expression(part)
        if expr is None:
            return None
        expressions.append(expr)
    return expressions


def _normalize_math_answer_text(answer: str) -> str:
    text = str(answer).strip()
    if not text:
        return ""
    fenced = re.fullmatch(r"```(?:text|latex)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    boxed_expr = _extract_boxed_expression(text)
    if boxed_expr is not None:
        text = boxed_expr.strip()
    answer_match = re.search(r"(?is)(?:final answer|answer)\s*[:=]?\s*(.+)$", text)
    if answer_match:
        text = answer_match.group(1).strip()
    text = text.replace("$", "").strip()
    return text


def _parse_math_expression(expr_text: str) -> sp.Expr | None:
    stripped = str(expr_text).strip()
    if not stripped:
        return None
    try:
        return parse_latex(stripped)
    except Exception:
        pass

    transformed = _latexish_to_sympy_expr(expr_text)
    if not transformed:
        return None
    try:
        return parse_expr(
            transformed,
            local_dict={"pi": sp.pi, "e": sp.E, "sqrt": sp.sqrt, "factorial": sp.factorial},
            transformations=standard_transformations + (implicit_multiplication_application,),
            evaluate=True,
        )
    except Exception:
        return None


def _latexish_to_sympy_expr(text: str) -> str:
    expr = str(text).strip()
    expr = expr.replace("\\left", "").replace("\\right", "")
    expr = expr.replace("\\cdot", "*").replace("\\times", "*")
    expr = expr.replace("\\pi", "pi")
    expr = re.sub(r"√\s*([A-Za-z0-9]+)", r"sqrt(\1)", expr)
    expr = expr.replace("^", "**")
    expr = _replace_latex_frac(expr)
    expr = _replace_latex_sqrt(expr)
    expr = re.sub(r"(\d+)\s*!", r"factorial(\1)", expr)
    expr = re.sub(r"\s+", "", expr)
    return expr


def _replace_latex_frac(expr: str) -> str:
    pattern = r"\\frac\s*\{"
    while True:
        match = re.search(pattern, expr)
        if match is None:
            return expr
        start = match.start()
        num_start = match.end() - 1
        numerator, num_end = _extract_braced(expr, num_start)
        if numerator is None:
            return expr
        if num_end >= len(expr) or expr[num_end] != "{":
            return expr
        denominator, den_end = _extract_braced(expr, num_end)
        if denominator is None:
            return expr
        replacement = f"(({numerator})/({denominator}))"
        expr = expr[:start] + replacement + expr[den_end:]


def _replace_latex_sqrt(expr: str) -> str:
    pattern = r"\\sqrt\s*\{"
    while True:
        match = re.search(pattern, expr)
        if match is None:
            return expr
        start = match.start()
        inner_start = match.end() - 1
        inner, inner_end = _extract_braced(expr, inner_start)
        if inner is None:
            return expr
        replacement = f"sqrt({inner})"
        expr = expr[:start] + replacement + expr[inner_end:]


def _extract_braced(text: str, start_idx: int) -> tuple[str | None, int]:
    if start_idx >= len(text) or text[start_idx] != "{":
        return None, start_idx
    depth = 0
    for idx in range(start_idx, len(text)):
        char = text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start_idx + 1 : idx], idx + 1
    return None, start_idx


def _extract_boxed_expression(text: str) -> str | None:
    marker = "\\boxed{"
    start = text.rfind(marker)
    if start == -1:
        return None
    inner, _end = _extract_braced(text, start + len("\\boxed"))
    return inner


def _format_mbpp_assert_failure(error_message: str) -> str:
    return error_message.strip() or "Assertion failed."


def _build_mbpp_harness(code: str, example: Any) -> str:
    setup_code = str(getattr(example, "test_setup_code", "") or "")
    test_list = list(getattr(example, "test_list", []) or [])
    challenge_test_list = list(getattr(example, "challenge_test_list", []) or [])
    all_tests = test_list + challenge_test_list

    harness_parts = [
        code.rstrip(),
        "",
        setup_code.rstrip(),
        "",
        "MBPP_TEST_RESULTS = []",
        "MBPP_FAILED_DETAILS = []",
        "MBPP_PASSED = True",
        "",
    ]

    for idx, test_line in enumerate(all_tests):
        test_literal = repr(str(test_line))
        harness_parts.extend(
            [
                "try:",
                f"    {test_line}",
                f"    MBPP_TEST_RESULTS.append({{'index': {idx}, 'passed': True, 'test': {test_literal}}})",
                "except Exception as e:",
                "    MBPP_PASSED = False",
                f"    MBPP_TEST_RESULTS.append({{'index': {idx}, 'passed': False, 'test': {test_literal}, 'error': str(e)}})",
                f"    MBPP_FAILED_DETAILS.append({{'index': {idx}, 'test': {test_literal}, 'error': str(e)}})",
                "",
            ]
        )

    return "\n".join(harness_parts).strip() + "\n"


def _extract_python_code(raw_response: str) -> str:
    text = str(raw_response).strip()
    fence_match = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        return fence_match.group(1).strip()
    return text


def _extract_mbpp_function_name(test_list: list[str], challenge_test_list: list[str]) -> str:
    all_tests = list(test_list) + list(challenge_test_list)
    for test in all_tests:
        match = re.search(r"assert\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", str(test))
        if match:
            return match.group(1)
    return ""


def _extract_mbpp_signature_line(canonical_solution: str) -> str:
    for line in str(canonical_solution).splitlines():
        stripped = line.strip()
        if stripped.startswith("def "):
            return stripped
    return ""


def _stable_test_id(*parts: str) -> str:
    blob = "||".join(parts).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def _build_test_solver_response(
    *,
    model_name: str,
    example: Any,
    prompt: str,
    output_mode: str,
    task: PromptOptimizationDatasetTask | None,
) -> dspy.Prediction:
    task_for_stub = task or get_default_dataset_task()
    is_success_prompt = "TEST_STUB_PROMPT_" in prompt
    reasoning = (
        f"[TEST_STUB id={_stable_test_id(model_name, str(getattr(example, 'input', '')), prompt, output_mode)}] "
        "Deterministic stub reasoning."
    )
    answer = (
        task_for_stub.test_stub_answer(example)
        if is_success_prompt
        else task_for_stub.test_stub_failure_answer(example)
    )
    normalized_answer, normalized_reasoning = _normalize_prediction_fields(
        answer,
        reasoning,
        output_mode=output_mode,
    )
    return dspy.Prediction(answer=normalized_answer, reasoning=normalized_reasoning)


def _build_test_reflection_response(*, model_name: str, prompt: str) -> str:
    stub_id = _stable_test_id(model_name, prompt)
    if "<REFLECTION>" in prompt and "<IMPROVED_PROMPT>" in prompt:
        return (
            f"<REFLECTION>Deterministic test reflection id={stub_id}</REFLECTION>"
            f"<IMPROVED_PROMPT>TEST_STUB_PROMPT_{stub_id}</IMPROVED_PROMPT>"
        )
    if "Return only:\n<prompt>" in prompt or "Return only:\r\n<prompt>" in prompt:
        return f"<prompt>TEST_STUB_PROMPT_{stub_id}</prompt>"
    if "<merged_suggestion>" in prompt:
        return f"<merged_suggestion>Use TEST_STUB_PROMPT_{stub_id} so the generated output satisfies the failing tests.</merged_suggestion>"
    if "<suggestion>" in prompt and "Return only:" in prompt:
        return f"<suggestion>Use TEST_STUB_PROMPT_{stub_id} so the generated output satisfies the failing tests.</suggestion>"
    if "<IMPROVED_VARIABLE>" in prompt and "</IMPROVED_VARIABLE>" in prompt:
        return f"<IMPROVED_VARIABLE>TEST_STUB_PROMPT_{stub_id}</IMPROVED_VARIABLE>"
    if "Provide only the new parameter value or prompt text within one final ``` block." in prompt:
        return f"```\nTEST_STUB_PROMPT_{stub_id}\n```"
    if "Provide the new instructions within ``` blocks." in prompt:
        return f"```\nTEST_STUB_PROMPT_{stub_id}\n```"
    return f"[TEST_STUB_REFLECTION id={stub_id}]"


def _load_math_dataset_impl(seed: int = 0) -> DatasetSplits:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not TRAIN_JSONL.exists():
        print(f"[AIME] Local train dataset not found. Downloading to {TRAIN_JSONL}...")
        train_rows = []
        train_load_dataset = load_dataset("AI-MO/aimo-validation-aime", "default", split="train")
        for item in train_load_dataset:
            train_rows.append(
                {
                    "input": item["problem"],
                    "solution": item["solution"],
                    "answer": item["answer"],
                }
            )
        with TRAIN_JSONL.open("w", encoding="utf-8") as f:
            for row in train_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[AIME] Saved train dataset JSONL: {TRAIN_JSONL} ({len(train_rows)} rows)")
    else:
        print(f"[AIME] Using local train dataset JSONL: {TRAIN_JSONL}")

    if not TEST_JSONL.exists():
        print(f"[AIME] Local test dataset not found. Downloading to {TEST_JSONL}...")
        test_rows = []
        test_load_dataset = load_dataset("MathArena/aime_2025", "default", split="train")
        for item in test_load_dataset:
            test_rows.append(
                {
                    "input": item["problem"],
                    "answer": item["answer"],
                }
            )
        with TEST_JSONL.open("w", encoding="utf-8") as f:
            for row in test_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[AIME] Saved test dataset JSONL: {TEST_JSONL} ({len(test_rows)} rows)")
    else:
        print(f"[AIME] Using local test dataset JSONL: {TEST_JSONL}")

    train_split = []
    with TRAIN_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            train_split.append(
                dspy.Example(input=item["input"], solution=item["solution"], answer=item["answer"]).with_inputs(
                    "input"
                )
            )

    random.Random(seed).shuffle(train_split)

    test_split = []
    with TEST_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            test_split.append(dspy.Example(input=item["input"], answer=item["answer"]).with_inputs("input"))

    print(
        f"[AIME] Loaded local JSONL data: train_total={len(train_split)}, "
        f"val_total={len(train_split) - (len(train_split) // 2)}, test_total={len(test_split)}"
    )

    train_size = len(train_split)
    trainset = train_split[: train_size // 2]
    valset = train_split[train_size // 2 :]
    testset = test_split

    return DatasetSplits(trainset=trainset, valset=valset, testset=testset)


def load_extra_math_testset(dataset_name: str, cache_path: Path) -> list[Any]:
    if not cache_path.exists():
        print(f"[AIME] Extra math test dataset not found locally. Downloading {dataset_name} to {cache_path}...")
        rows = []
        raw_dataset = load_dataset(dataset_name, "default", split="train")
        for item in raw_dataset:
            rows.append(
                {
                    "input": item["problem"],
                    "answer": item["answer"],
                    "problem_id": item.get("problem_idx"),
                    "problem_type": item.get("problem_type", ""),
                }
            )
        with cache_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[AIME] Saved extra math test dataset JSONL: {cache_path} ({len(rows)} rows)")
    else:
        print(f"[AIME] Using local extra math test dataset JSONL: {cache_path}")

    dataset = []
    with cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            dataset.append(
                dspy.Example(
                    input=item["input"],
                    answer=item["answer"],
                    problem_id=item.get("problem_id"),
                    problem_type=item.get("problem_type", ""),
                ).with_inputs("input")
            )
    return dataset


def _evaluate_math_example(
    prompt: str,
    example: Any,
    *,
    use_solver_cache: bool | None = None,
    cache_extra: dict | None = None,
    lookup_cache_extra: dict | None = None,
) -> tuple[float, dict[str, Any]]:
    prediction = run_llm(
        example,
        prompt,
        use_solver_cache=use_solver_cache,
        cache_extra=cache_extra,
        lookup_cache_extra=lookup_cache_extra,
    )
    score, feedback = math_metric(example, prediction)
    side_info = {
        "score": score,
        "input": example.input,
        "prompt": prompt,
        "output": prediction.answer,
        "reasoning": getattr(prediction, "reasoning", ""),
        "execution_feedback": feedback,
    }
    return score, side_info


def _evaluate_math_dataset_impl(
    prompt,
    dataset,
    max_workers: int = 32,
    use_solver_cache: bool | None = None,
    pass_k: int = 1,
    return_stats: bool = False,
    cache_label: str | None = None,
):
    """Evaluate a prompt on a dataset via run_llm with parse-error resilience."""
    if pass_k < 1:
        raise ValueError(f"pass_k must be >= 1, got {pass_k}")

    total = len(dataset)
    if total == 0:
        return 0.0

    pass_score_sum = 0.0
    mean_score_sum = 0.0
    attempt_count = 0
    completed = 0

    def evaluate_one(example):
        scores = []
        for attempt_idx in range(pass_k):
            cache_extra = None
            lookup_cache_extra = None
            if pass_k > 1:
                cache_extra = {
                    "pass_k": pass_k,
                    "attempt_idx": attempt_idx,
                }
            if cache_label is not None or pass_k > 1:
                lookup_cache_extra = {
                    "pass_k": pass_k,
                    "attempt_idx": attempt_idx,
                }
            score, _side_info = _evaluate_math_example(
                prompt,
                example,
                use_solver_cache=use_solver_cache,
                cache_extra=cache_extra,
                lookup_cache_extra=lookup_cache_extra,
            )
            scores.append(score)
        return {
            "pass_score": float(any(score >= 1.0 for score in scores)),
            "mean_score": sum(scores) / len(scores),
            "attempt_count": len(scores),
        }

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(evaluate_one, example) for example in dataset]
        for future in as_completed(futures):
            result = future.result()
            pass_score_sum += result["pass_score"]
            mean_score_sum += result["mean_score"]
            attempt_count += result["attempt_count"]
            completed += 1

            if completed % 10 == 0 or completed == total:
                print(
                    f"[AIME] Eval progress: {completed}/{total} "
                    f"(pass@{pass_k}={pass_score_sum / completed:.2%}, "
                    f"mean@{pass_k}={mean_score_sum / completed:.2%})"
                )

    stats = {
        "pass_k": pass_k,
        "pass_score": pass_score_sum / total,
        "mean_score": mean_score_sum / total,
        "total_examples": total,
        "total_attempts": attempt_count,
    }
    if return_stats:
        return stats
    return stats["pass_score"]


class AIMEMathTask(PromptOptimizationDatasetTask):
    """Default shared dataset/evaluation backend for the AIME examples."""

    def load_splits(self, *, seed: int = 0) -> DatasetSplits:
        return _load_math_dataset_impl(seed=seed)

    def evaluate_example(
        self,
        prompt: str,
        example: Any,
        *,
        use_solver_cache: bool | None = None,
        cache_extra: dict | None = None,
        lookup_cache_extra: dict | None = None,
    ) -> tuple[float, dict[str, Any]]:
        return _evaluate_math_example(
            prompt,
            example,
            use_solver_cache=use_solver_cache,
            cache_extra=cache_extra,
            lookup_cache_extra=lookup_cache_extra,
        )

    def evaluate_dataset(
        self,
        prompt: str,
        dataset: list[Any],
        *,
        max_workers: int = 32,
        use_solver_cache: bool | None = None,
        pass_k: int = 1,
        return_stats: bool = False,
        cache_label: str | None = None,
    ):
        return _evaluate_math_dataset_impl(
            prompt,
            dataset,
            max_workers=max_workers,
            use_solver_cache=use_solver_cache,
            pass_k=pass_k,
            return_stats=return_stats,
            cache_label=cache_label,
        )

    def make_reflection_instruction(self, current_prompt: str, current_batch_feedback: str, current_batch_score: float) -> str:
        del current_batch_feedback, current_batch_score
        return current_prompt

    def textgrad_evaluation_instruction(self) -> str:
        return (
            "Below is an AIME-style math problem, the correct integer answer, and a reasoning trace with a final "
            "prediction from the language model. Critically evaluate the reasoning and final answer. Identify the "
            "most important mathematical mistakes, answer-format mistakes, or missed problem structure. Give concise, "
            "actionable feedback that would help improve the system prompt for future AIME-style problems."
        )

    def generate_model_output(
        self,
        example: Any,
        prompt: str,
        *,
        use_solver_cache: bool | None = None,
        cache_extra: dict | None = None,
        lookup_cache_extra: dict | None = None,
    ) -> tuple[str, str]:
        prediction = run_llm(
            example,
            prompt,
            use_solver_cache=use_solver_cache,
            cache_extra=cache_extra,
            lookup_cache_extra=lookup_cache_extra,
        )
        return str(getattr(prediction, "answer", "")), str(getattr(prediction, "reasoning", ""))

    def test_stub_answer(self, example: Any) -> str:
        return str(getattr(example, "answer", "0"))

    def test_stub_failure_answer(self, example: Any) -> str:
        answer = int(getattr(example, "answer", "0"))
        return str(answer + 1)


class MBPPTask(PromptOptimizationDatasetTask):
    """Shared dataset/evaluation backend for MBPP code-generation runs."""

    _HF_SPLIT_MAP = {
        "train": "train",
        "val": "validation",
        "validation": "validation",
        "test": "test",
    }
    _LOCAL_SPLIT_MAP = {
        "train": "train",
        "val": "val",
        "validation": "val",
        "test": "test",
    }

    def __init__(
        self,
        data_dir: Path | None = None,
        *,
        source: str = "huggingface",
        hf_dataset: str = "google-research-datasets/mbpp",
        hf_config: str | None = "full",
    ):
        self.data_dir = Path(data_dir) if data_dir is not None else None
        self.source = source.strip().lower()
        self.hf_dataset = hf_dataset
        self.hf_config = hf_config
        if self.source not in {"huggingface", "local"}:
            raise ValueError(f"Unsupported MBPP source: {source!r}")

    def load_splits(self, *, seed: int = 0) -> DatasetSplits:
        del seed
        return DatasetSplits(
            trainset=self._load_split("train"),
            valset=self._load_split("val"),
            testset=self._load_split("test"),
        )

    def _load_split(self, split: str) -> list[Any]:
        if self.source == "huggingface":
            return self._load_hf_split(split)
        return self._load_local_split(split)

    def _load_local_split(self, split: str) -> list[Any]:
        if self.data_dir is None:
            raise ValueError("mbpp_data_dir must be provided when AIME_MBPP_SOURCE=local")
        split_name = self._LOCAL_SPLIT_MAP[split]
        split_path = self.data_dir / f"{split_name}.jsonl"
        if not split_path.exists():
            raise FileNotFoundError(f"Missing MBPP split file: {split_path}")

        examples: list[Any] = []
        with split_path.open("r", encoding="utf-8") as f:
            for line in f:
                payload = json.loads(line)
                examples.append(self._payload_to_example(payload))
        return examples

    def _load_hf_split(self, split: str) -> list[Any]:
        hf_split = self._HF_SPLIT_MAP[split]
        dataset = load_dataset(self.hf_dataset, self.hf_config, split=hf_split)
        return [self._payload_to_example(dict(item)) for item in dataset]

    def _payload_to_example(self, payload: dict[str, Any]) -> Any:
        labels = payload.get("labels", {}) or {}
        test_list = list(payload.get("test_list", []) or [])
        challenge_test_list = list(
            payload.get("challenge_test_list", []) or labels.get("challenge_test_list", []) or []
        )
        canonical_solution = str(
            payload.get("canonical_solution", payload.get("code", "")) or ""
        )
        content = str(payload.get("content", payload.get("text", "")) or "")
        test_setup_code = str(
            payload.get("test_setup_code", labels.get("test_setup_code", "")) or ""
        )
        raw_problem_id = payload.get("id", payload.get("task_id"))
        if raw_problem_id is None:
            raise ValueError(f"MBPP example missing id/task_id: {payload!r}")
        problem_id = int(raw_problem_id)
        function_name = _extract_mbpp_function_name(test_list, challenge_test_list)
        signature_line = _extract_mbpp_signature_line(canonical_solution)
        input_text = content
        if function_name:
            input_text = (
                f"{content}\n\n"
                f"Required function name: {function_name}\n"
                "Your code must define exactly this function name."
            )
        if signature_line:
            input_text = (
                f"{input_text}\n"
                f'Function signature: "{signature_line}"'
            )
        return dspy.Example(
            input=input_text,
            answer=canonical_solution,
            problem_id=problem_id,
            content=content,
            function_name=function_name,
            signature_line=signature_line,
            canonical_solution=canonical_solution,
            test_list=test_list,
            challenge_test_list=challenge_test_list,
            test_setup_code=test_setup_code,
        ).with_inputs("input")

    def evaluate_example(
        self,
        prompt: str,
        example: Any,
        *,
        use_solver_cache: bool | None = None,
        cache_extra: dict | None = None,
        lookup_cache_extra: dict | None = None,
    ) -> tuple[float, dict[str, Any]]:
        generated_code, reasoning = self.generate_model_output(
            example,
            prompt,
            use_solver_cache=use_solver_cache,
            cache_extra=cache_extra,
            lookup_cache_extra=lookup_cache_extra,
        )
        harness = _build_mbpp_harness(generated_code, example)
        result = execute_code(
            code=harness,
            timeout=10,
            mode=ExecutionMode.SUBPROCESS,
            capture_variables=["MBPP_PASSED", "MBPP_TEST_RESULTS", "MBPP_FAILED_DETAILS"],
        )
        variables = result.variables
        test_results = list(variables.get("MBPP_TEST_RESULTS", []) or [])
        failed_details = list(variables.get("MBPP_FAILED_DETAILS", []) or [])
        passed = bool(result.success and variables.get("MBPP_PASSED") is True)
        total_tests = len(test_results)
        num_failed = len(failed_details)
        score = 1.0 if passed else 0.0

        if not result.success:
            feedback = (
                "Execution failed before the test suite completed. "
                f"Error: {result.error or 'unknown error'}. "
                "Fix syntax/runtime issues and ensure the code defines the required function."
            )
        elif passed:
            feedback = f"Passed all {total_tests} tests. Preserve the working structure and correctness."
        else:
            first_failure = failed_details[0] if failed_details else {}
            feedback = (
                f"Failed {num_failed} / {total_tests} tests. "
                f"First failing test: {first_failure.get('test', 'n/a')}. "
                f"Failure detail: {_format_mbpp_assert_failure(str(first_failure.get('error', '')))}. "
                "Use the failing assertions, runtime output, and traceback to repair the code."
            )

        side_info = {
            "score": score,
            "input": getattr(example, "content", example.input),
            "prompt": prompt,
            "output": generated_code,
            "reasoning": reasoning,
            "execution_feedback": feedback,
            "problem_id": int(getattr(example, "problem_id")),
            "passed": passed,
            "total_tests": total_tests,
            "failed_tests": num_failed,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "error": result.error,
            "traceback": result.traceback,
            "execution_time": result.execution_time,
            "test_results": test_results,
            "failed_details": failed_details,
            "public_test_list": list(getattr(example, "test_list", []) or []),
            "challenge_test_list": list(getattr(example, "challenge_test_list", []) or []),
        }
        return score, side_info

    def evaluate_dataset(
        self,
        prompt: str,
        dataset: list[Any],
        *,
        max_workers: int = 32,
        use_solver_cache: bool | None = None,
        pass_k: int = 1,
        return_stats: bool = False,
        cache_label: str | None = None,
    ):
        if pass_k < 1:
            raise ValueError(f"pass_k must be >= 1, got {pass_k}")

        total = len(dataset)
        if total == 0:
            return 0.0

        pass_score_sum = 0.0
        mean_score_sum = 0.0
        attempt_count = 0
        completed = 0

        def evaluate_one(example):
            scores = []
            for attempt_idx in range(pass_k):
                # Mirror the AIME math path: give each pass@k attempt a distinct
                # cache key so that, when the solver cache is enabled, attempts do
                # not collapse onto a single cached prediction (which would turn
                # pass@k into pass@1).
                cache_extra = None
                lookup_cache_extra = None
                if pass_k > 1:
                    cache_extra = {"pass_k": pass_k, "attempt_idx": attempt_idx}
                if cache_label is not None or pass_k > 1:
                    lookup_cache_extra = {"pass_k": pass_k, "attempt_idx": attempt_idx}
                score, _ = self.evaluate_example(
                    prompt,
                    example,
                    use_solver_cache=use_solver_cache,
                    cache_extra=cache_extra,
                    lookup_cache_extra=lookup_cache_extra,
                )
                scores.append(score)
            return {
                "pass_score": float(any(score >= 1.0 for score in scores)),
                "mean_score": sum(scores) / len(scores),
                "attempt_count": len(scores),
            }

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(evaluate_one, example) for example in dataset]
            for future in as_completed(futures):
                result = future.result()
                pass_score_sum += result["pass_score"]
                mean_score_sum += result["mean_score"]
                attempt_count += result["attempt_count"]
                completed += 1
                if completed % 10 == 0 or completed == total:
                    print(
                        f"[MBPP] Eval progress: {completed}/{total} "
                        f"(pass@{pass_k}={pass_score_sum / completed:.2%}, "
                        f"mean@{pass_k}={mean_score_sum / completed:.2%})"
                    )

        stats = {
            "pass_k": pass_k,
            "pass_score": pass_score_sum / total,
            "mean_score": mean_score_sum / total,
            "total_examples": total,
            "total_attempts": attempt_count,
        }
        if return_stats:
            return stats
        return stats["pass_score"]

    def make_reflection_instruction(self, current_prompt: str, current_batch_feedback: str, current_batch_score: float) -> str:
        del current_batch_score
        return (
            f"{current_prompt.strip()}\n\n"
            "You are optimizing a prompt that asks a model to generate Python code for MBPP tasks. "
            "The evaluation feedback below includes test assertions, pass/fail outcomes, runtime errors, stdout/stderr, "
            "and traceback snippets. Focus on changes that improve code correctness, function signatures, imports, "
            "and robustness to the provided tests.\n\n"
            f"Recent evaluation feedback:\n{current_batch_feedback.strip()}"
        )

    def textgrad_evaluation_instruction(self) -> str:
        return (
            "You are analyzing Python code generated for an MBPP programming task together with unit test results. "
            "The code may be tested with harder hidden tests, so do not only check whether it passes the provided tests. "
            "Investigate correctness, required function signatures, imports, runtime behavior, and likely failure modes on "
            "slightly harder cases. For each failed test, explain why the current implementation does not produce the expected "
            "behavior. Do not provide a revised implementation. Give very concise, actionable feedback that would help improve "
            "the system prompt for future MBPP code-generation tasks."
        )

    def generate_model_output(
        self,
        example: Any,
        prompt: str,
        *,
        use_solver_cache: bool | None = None,
        cache_extra: dict | None = None,
        lookup_cache_extra: dict | None = None,
    ) -> tuple[str, str]:
        prediction = run_llm(
            example,
            prompt,
            use_solver_cache=use_solver_cache,
            cache_extra=cache_extra,
            lookup_cache_extra=lookup_cache_extra,
        )
        return str(getattr(prediction, "answer", "")), str(getattr(prediction, "reasoning", ""))

    def test_stub_answer(self, example: Any) -> str:
        return (
            "```python\n"
            f"{str(getattr(example, 'canonical_solution', '')).strip()}\n"
            "```"
        )

    def test_stub_failure_answer(self, example: Any) -> str:
        return (
            "```python\n"
            "def __test_stub_failure__(*args, **kwargs):\n"
            "    return None\n"
            "```"
        )


def build_dataset_task(
    dataset_name: str,
    *,
    mbpp_source: str = "huggingface",
    mbpp_hf_dataset: str = "google-research-datasets/mbpp",
    mbpp_hf_config: str | None = "full",
    mbpp_data_dir: Path | None = None,
) -> PromptOptimizationDatasetTask:
    dataset_name = dataset_name.strip().lower()
    if dataset_name == "aime":
        return AIMEMathTask()
    if dataset_name == "mbpp":
        return MBPPTask(
            mbpp_data_dir,
            source=mbpp_source,
            hf_dataset=mbpp_hf_dataset,
            hf_config=mbpp_hf_config,
        )
    raise ValueError(f"Unsupported dataset_name: {dataset_name!r}")


def set_default_dataset_task(task: PromptOptimizationDatasetTask) -> None:
    global _DEFAULT_DATASET_TASK
    _DEFAULT_DATASET_TASK = task


def get_default_dataset_task() -> PromptOptimizationDatasetTask:
    global _DEFAULT_DATASET_TASK
    if _DEFAULT_DATASET_TASK is None:
        _DEFAULT_DATASET_TASK = AIMEMathTask()
    return _DEFAULT_DATASET_TASK


def load_math_dataset(seed: int = 0):
    splits = get_default_dataset_task().load_splits(seed=seed)
    return splits.trainset, splits.valset, splits.testset


def evaluate_on_dataset(
    prompt,
    dataset,
    max_workers: int = 32,
    use_solver_cache: bool | None = None,
    pass_k: int = 1,
    return_stats: bool = False,
    cache_label: str | None = None,
):
    return get_default_dataset_task().evaluate_dataset(
        prompt,
        dataset,
        max_workers=max_workers,
        use_solver_cache=use_solver_cache,
        pass_k=pass_k,
        return_stats=return_stats,
        cache_label=cache_label,
    )


def generate_evaluation_plot(
    run_dir: Path,
    baseline_score: float,
    optimized_score: float,
    pass_k: int,
    baseline_mean_score: float | None = None,
    optimized_mean_score: float | None = None,
    output_stem: str = "latest_run_result_plot",
) -> dict:
    """Parse a run's logs and write the final evaluation plot plus summary JSON."""
    run_dir = Path(run_dir)
    summary = _build_evaluation_plot_summary(
        run_dir=run_dir,
        baseline_score=baseline_score,
        optimized_score=optimized_score,
        pass_k=pass_k,
        baseline_mean_score=baseline_mean_score,
        optimized_mean_score=optimized_mean_score,
    )
    summary_path = run_dir / f"{output_stem}_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    png_path = run_dir / f"{output_stem}.png"
    try:
        _write_evaluation_plot_png(summary, png_path)
        summary["plot_path"] = str(png_path)
    except Exception as exc:
        svg_path = run_dir / f"{output_stem}.svg"
        _write_evaluation_plot_svg(summary, svg_path)
        summary["plot_path"] = str(svg_path)
        summary["plot_warning"] = f"matplotlib plot failed; wrote SVG fallback: {type(exc).__name__}: {exc}"

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _build_evaluation_plot_summary(
    run_dir: Path,
    baseline_score: float,
    optimized_score: float,
    pass_k: int,
    baseline_mean_score: float | None = None,
    optimized_mean_score: float | None = None,
) -> dict:
    log_path = run_dir / "run_log.txt"
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    number = r"([0-9]+(?:\.[0-9]+)?)"
    optimizer_backend = "gepa"
    optimizer_label = "GEPA"

    candidate_points = []
    for match in re.finditer(
        rf"Iteration (\d+): Valset score for new program: {number} \(coverage (\d+) / (\d+)\)",
        text,
    ):
        candidate_points.append(
            {
                "iteration": int(match.group(1)),
                "score": float(match.group(2)),
                "coverage": int(match.group(3)),
                "total": int(match.group(4)),
            }
        )

    best_events = []
    for match in re.finditer(rf"Iteration (\d+): Found a better program on the valset with score {number}", text):
        best_events.append({"iteration": int(match.group(1)), "score": float(match.group(2))})

    pareto_points = []
    for match in re.finditer(rf"Iteration (\d+): Valset pareto front aggregate score: {number}", text):
        pareto_points.append({"iteration": int(match.group(1)), "score": float(match.group(2))})

    candidate_points = _dedupe_plot_points(candidate_points, ("iteration", "score", "coverage", "total"))
    best_events = _dedupe_plot_points(best_events, ("iteration", "score"))
    pareto_points = _dedupe_plot_points(pareto_points, ("iteration", "score"))

    if not candidate_points:
        sample_weighted_summary = _load_sample_weighted_gepa_plot_summary(run_dir)
        if sample_weighted_summary is not None:
            optimizer_backend = "sample_weighted_gepa"
            optimizer_label = "Sample Weighted GEPA"
            candidate_points = sample_weighted_summary["candidate_points"]
            best_events = sample_weighted_summary["best_events"]
            pareto_points = []
    if not candidate_points:
        textgrad_summary = _load_textgrad_plot_summary(run_dir)
        if textgrad_summary is not None:
            optimizer_backend = "textgrad"
            optimizer_label = f"TextGrad ({textgrad_summary['algorithm']})"
            candidate_points = textgrad_summary["candidate_points"]
            best_events = textgrad_summary["best_events"]
            pareto_points = []
    if not candidate_points:
        parent_reflection_summary = _load_parent_reflection_gepa_plot_summary(run_dir)
        if parent_reflection_summary is not None:
            optimizer_backend = "parent_reflection_gepa"
            optimizer_label = "Parent Reflection GEPA"
            candidate_points = parent_reflection_summary["candidate_points"]
            best_events = parent_reflection_summary["best_events"]
            pareto_points = []
    if not candidate_points:
        trace2skill_summary = _load_trace2skill_plot_summary(run_dir)
        if trace2skill_summary is not None:
            optimizer_backend = "trace2skill_baseline"
            optimizer_label = "Trace2Skill Baseline"
            candidate_points = trace2skill_summary["candidate_points"]
            best_events = trace2skill_summary["best_events"]
            pareto_points = []

    candidates_path = run_dir / "candidates.json"
    if candidates_path.exists():
        num_candidates = len(json.loads(candidates_path.read_text(encoding="utf-8")))
    else:
        num_candidates = len(candidate_points)

    return {
        "run_dir": str(run_dir),
        "optimizer_backend": optimizer_backend,
        "optimizer_label": optimizer_label,
        "num_candidates": num_candidates,
        "num_candidate_val_evals": len(candidate_points),
        "candidate_points": candidate_points,
        "best_events": best_events,
        "pareto_points": pareto_points,
        "final_best_val_score": max((point["score"] for point in candidate_points), default=None),
        "final_pareto_val_score": pareto_points[-1]["score"] if pareto_points else None,
        "test_results": {
            "pass_k": pass_k,
            "baseline_score": baseline_score,
            "optimized_score": optimized_score,
            "improvement": optimized_score - baseline_score,
            "baseline_mean_score": baseline_mean_score,
            "optimized_mean_score": optimized_mean_score,
            "mean_improvement": (
                optimized_mean_score - baseline_mean_score
                if baseline_mean_score is not None and optimized_mean_score is not None
                else None
            ),
        },
    }


def _load_textgrad_plot_summary(run_dir: Path) -> dict | None:
    summary_paths = sorted(
        path
        for path in run_dir.glob("textgrad_*_summary.json")
        if not path.name.endswith("_result_plot_summary.json")
    )
    if not summary_paths:
        return None

    payload = json.loads(summary_paths[0].read_text(encoding="utf-8"))
    history = payload.get("prompt_history", [])
    if not history:
        return None

    candidate_points = []
    best_events = []
    best_score = None
    valset_total = 45

    for entry in history:
        val_score = entry.get("val_score")
        if val_score is None:
            continue
        score = float(val_score)
        point = {
            "iteration": int(entry.get("step_idx", 0)) + 1,
            "score": score,
            "coverage": valset_total,
            "total": valset_total,
        }
        candidate_points.append(point)
        if best_score is None or score > best_score:
            best_score = score
            best_events.append({"iteration": point["iteration"], "score": score})

    if not candidate_points:
        return None

    return {
        "algorithm": str(payload.get("algorithm", "textgrad")).upper(),
        "candidate_points": candidate_points,
        "best_events": best_events,
    }


def _load_sample_weighted_gepa_plot_summary(run_dir: Path) -> dict | None:
    summary_path = run_dir / "sample_weighted_gepa_summary.json"
    if not summary_path.exists():
        return None

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    history = load_sample_weighted_gepa_candidate_points(payload)
    if not history:
        return None

    valset_total = infer_valset_total_from_summary(payload)
    candidate_points = [
        {
            "iteration": int(entry.get("iteration", idx)),
            "score": float(entry["score"]),
            "coverage": valset_total,
            "total": valset_total,
        }
        for idx, entry in enumerate(history, start=1)
    ]
    return {
        "candidate_points": candidate_points,
        "best_events": build_best_events(candidate_points),
    }


def _load_parent_reflection_gepa_plot_summary(run_dir: Path) -> dict | None:
    summary_path = run_dir / "parent_reflection_gepa_summary.json"
    if not summary_path.exists():
        return None

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    history = payload.get("candidate_points", [])
    if not history:
        return None

    valset_total = int(payload.get("valset_size", 45))
    candidate_points = []
    best_events = []
    best_score = None

    for idx, entry in enumerate(history, start=1):
        if "score" not in entry:
            continue
        score = float(entry["score"])
        point = {
            "iteration": int(entry.get("iteration", idx)),
            "score": score,
            "coverage": valset_total,
            "total": valset_total,
        }
        candidate_points.append(point)
        if best_score is None or score > best_score:
            best_score = score
            best_events.append({"iteration": point["iteration"], "score": score})

    if not candidate_points:
        return None

    return {
        "candidate_points": candidate_points,
        "best_events": build_best_events(candidate_points),
    }


def _load_trace2skill_plot_summary(run_dir: Path) -> dict | None:
    summary_path = run_dir / "trace2skill_baseline" / "summary.json"
    if not summary_path.exists():
        return None

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    history = payload.get("iterations", [])
    if not history:
        return None

    trainset_total = int(payload.get("trainset_size", 45))
    candidate_points = []
    best_events = []
    best_score = None

    for entry in history:
        if "val_score" not in entry:
            continue
        score = float(entry["val_score"])
        point = {
            "iteration": int(entry.get("iteration", 0)),
            "score": score,
            "coverage": trainset_total,
            "total": trainset_total,
        }
        candidate_points.append(point)
        if best_score is None or score > best_score:
            best_score = score
            best_events.append({"iteration": point["iteration"], "score": score})

    if not candidate_points:
        return None

    return {
        "candidate_points": candidate_points,
        "best_events": best_events,
    }


def _dedupe_plot_points(points: list[dict], keys: tuple[str, ...]) -> list[dict]:
    seen = set()
    out = []
    for point in points:
        key = tuple(point[name] for name in keys)
        if key in seen:
            continue
        seen.add(key)
        out.append(point)
    return out


def _running_best_points(candidate_points: list[dict]) -> list[dict]:
    out = []
    best = None
    for point in candidate_points:
        score = point["score"]
        if best is None or score > best:
            best = score
        out.append({"iteration": point["iteration"], "score": best})
    return out


def _safe_legend(ax, **kwargs) -> None:
    handles, labels = ax.get_legend_handles_labels()
    visible = [(handle, label) for handle, label in zip(handles, labels, strict=True) if label and not label.startswith("_")]
    if not visible:
        return
    safe_handles, safe_labels = zip(*visible, strict=True)
    ax.legend(safe_handles, safe_labels, **kwargs)


def _score_axis_limits(candidate_points: list[dict], running_best_points: list[dict], pareto_points: list[dict]) -> tuple[float, float]:
    scores = [point["score"] for point in candidate_points]
    scores.extend(point["score"] for point in running_best_points)
    scores.extend(point["score"] for point in pareto_points)
    if not scores:
        return 0.50, 0.93

    min_score = min(scores)
    max_score = max(scores)
    padding = max(0.03, 0.08 * max(max_score - min_score, 0.1))
    y_min = max(0.0, min_score - padding)
    y_max = min(1.05, max_score + padding)
    if y_max - y_min < 0.1:
        center = (y_min + y_max) / 2
        y_min = max(0.0, center - 0.05)
        y_max = min(1.05, center + 0.05)
    return y_min, y_max


def _fail_axis_limit(candidate_points: list[dict]) -> float:
    if not candidate_points:
        return 20.0
    max_fail = max(point["total"] * (1.0 - point["score"]) for point in candidate_points)
    return max(5.0, max_fail + 2.0)


def _write_evaluation_plot_png(summary: dict, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    candidate_points = summary["candidate_points"]
    running_best_points = _running_best_points(candidate_points)
    pareto_points = summary["pareto_points"]
    best_events = summary["best_events"]
    test_results = summary["test_results"]
    optimizer_label = summary.get("optimizer_label", "GEPA")
    baseline_mean = test_results.get("baseline_mean_score")
    optimized_mean = test_results.get("optimized_mean_score")
    mean_improvement = test_results.get("mean_improvement")
    score_y_min, score_y_max = _score_axis_limits(candidate_points, running_best_points, pareto_points)
    fail_y_max = _fail_axis_limit(candidate_points)

    fig = plt.figure(figsize=(12.5, 8.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[2.1, 1.0], height_ratios=[1.25, 1.0])
    ax = fig.add_subplot(grid[0, 0])
    ax_fail = fig.add_subplot(grid[1, 0], sharex=ax)
    ax_test = fig.add_subplot(grid[:, 1])

    if candidate_points:
        xs = [point["iteration"] for point in candidate_points]
        ys = [point["score"] for point in candidate_points]
        ax.scatter(xs, ys, s=52, color="#355070", alpha=0.78, label="candidate val score", zorder=3)
        ax.plot(xs, ys, color="#355070", alpha=0.25, linewidth=1.2)

    if running_best_points:
        xs = [point["iteration"] for point in running_best_points]
        ys = [point["score"] for point in running_best_points]
        ax.step(xs, ys, where="post", color="#2a9d8f", linewidth=2.5, label="best val score so far")

    if pareto_points:
        xs = [point["iteration"] for point in pareto_points]
        ys = [point["score"] for point in pareto_points]
        ax.plot(
            xs,
            ys,
            color="#f4a261",
            marker="D",
            markersize=5.5,
            linewidth=2.0,
            linestyle=":",
            label="pareto front aggregate",
        )

    for point in best_events:
        ax.axvline(point["iteration"], color="#2a9d8f", alpha=0.16, linewidth=1.2)

    ax.set_title(f"AIME {optimizer_label} Run: Optimization Progress")
    ax.set_ylabel("validation score")
    ax.set_ylim(score_y_min, score_y_max)
    ax.grid(True, alpha=0.25)
    _safe_legend(ax, loc="lower right")

    if candidate_points:
        xs = [point["iteration"] for point in candidate_points]
        fails = [point["total"] * (1.0 - point["score"]) for point in candidate_points]
        ax_fail.bar(xs, fails, width=1.2, color="#e76f51", alpha=0.55, label="failed val examples")
        for x_value, fail_count in zip(xs, fails):
            ax_fail.text(
                x_value,
                fail_count + 0.25,
                str(int(round(fail_count))),
                ha="center",
                va="bottom",
                fontsize=8,
                color="#6f2a1f",
            )
    ax_fail.set_xlabel("iteration")
    ax_fail.set_ylabel("failed / 45")
    ax_fail.set_ylim(0, fail_y_max)
    ax_fail.grid(True, axis="y", alpha=0.25)
    _safe_legend(ax_fail, loc="upper right")

    labels = ["Baseline", "Optimized"]
    values = [test_results["baseline_score"], test_results["optimized_score"]]
    colors = ["#6c757d", "#2a9d8f"]
    bars = ax_test.bar(labels, values, color=colors, width=0.58)
    ax_test.set_title(f"Final AIME Test pass@{test_results['pass_k']}")
    ax_test.set_ylim(0, max(0.75, max(values) + 0.15))
    ax_test.set_ylabel("score")
    ax_test.grid(True, axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax_test.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.018,
            f"{value:.2%}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )
    ax_test.text(
        0.5,
        0.08,
        f"pass +{test_results['improvement']:.2%}",
        transform=ax_test.transAxes,
        ha="center",
        va="center",
        fontsize=13,
        color="#2a9d8f",
        fontweight="bold",
    )
    if baseline_mean is not None and optimized_mean is not None and mean_improvement is not None:
        ax_test.text(
            0.5,
            0.035,
            f"mean: {baseline_mean:.2%} -> {optimized_mean:.2%} ({mean_improvement:+.2%})",
            transform=ax_test.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="#444444",
        )

    fig.suptitle(f"Run {Path(summary['run_dir']).name}", fontsize=14, fontweight="bold")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _write_evaluation_plot_svg(summary: dict, output_path: Path) -> None:
    candidate_points = summary["candidate_points"]
    running_best_points = _running_best_points(candidate_points)
    pareto_points = summary["pareto_points"]
    test_results = summary["test_results"]
    optimizer_label = summary.get("optimizer_label", "GEPA")
    baseline = test_results["baseline_score"]
    optimized = test_results["optimized_score"]
    improvement = test_results["improvement"]
    baseline_mean = test_results.get("baseline_mean_score")
    optimized_mean = test_results.get("optimized_mean_score")
    mean_improvement = test_results.get("mean_improvement")
    score_y_min, score_y_max = _score_axis_limits(candidate_points, running_best_points, pareto_points)
    fail_y_max = _fail_axis_limit(candidate_points)
    mean_line = ""
    if baseline_mean is not None and optimized_mean is not None and mean_improvement is not None:
        mean_line = (
            f'<text x="695" y="512" text-anchor="middle" font-family="Arial, sans-serif" '
            f'font-size="13" fill="#444">mean: {baseline_mean:.2%} -> {optimized_mean:.2%} '
            f'({mean_improvement:+.2%})</text>'
        )
    best_val = summary["final_best_val_score"] or 0.0
    pareto = summary["final_pareto_val_score"] or 0.0
    baseline_height = 260 * baseline
    optimized_height = 260 * optimized

    chart_x = 58
    chart_y = 118
    chart_w = 430
    chart_h = 190
    fail_y = 360
    fail_h = 105
    iterations = [point["iteration"] for point in candidate_points] or [0, 1]
    min_iter = min(iterations)
    max_iter = max(iterations)
    if min_iter == max_iter:
        max_iter = min_iter + 1

    def x_coord(iteration: int) -> float:
        return chart_x + ((iteration - min_iter) / (max_iter - min_iter)) * chart_w

    def y_coord(score: float) -> float:
        if score_y_max == score_y_min:
            return chart_y + chart_h / 2
        return chart_y + ((score_y_max - score) / (score_y_max - score_y_min)) * chart_h

    def fail_height(fail_count: float) -> float:
        if fail_y_max <= 0:
            return 0.0
        return (fail_count / fail_y_max) * fail_h

    def points_attr(points: list[dict]) -> str:
        return " ".join(f"{x_coord(point['iteration']):.1f},{y_coord(point['score']):.1f}" for point in points)

    candidate_circles = "\n".join(
        f'  <circle cx="{x_coord(point["iteration"]):.1f}" cy="{y_coord(point["score"]):.1f}" r="4.8" fill="#355070" opacity="0.82"/>'
        for point in candidate_points
    )
    best_polyline = (
        f'  <polyline points="{points_attr(running_best_points)}" fill="none" stroke="#2a9d8f" '
        'stroke-width="3" stroke-linejoin="round"/>'
        if running_best_points
        else ""
    )
    pareto_polyline = (
        f'  <polyline points="{points_attr(pareto_points)}" fill="none" stroke="#f4a261" '
        'stroke-width="3" stroke-dasharray="5 5" stroke-linejoin="round"/>'
        if pareto_points
        else ""
    )
    pareto_dots = "\n".join(
        f'  <rect x="{x_coord(point["iteration"]) - 4.5:.1f}" y="{y_coord(point["score"]) - 4.5:.1f}" width="9" height="9" fill="#f4a261" transform="rotate(45 {x_coord(point["iteration"]):.1f} {y_coord(point["score"]):.1f})"/>'
        for point in pareto_points
    )
    fail_bars = "\n".join(
        f'  <rect x="{x_coord(point["iteration"]) - 5:.1f}" y="{fail_y + fail_h - fail_height(point["total"] * (1.0 - point["score"])):.1f}" width="10" height="{fail_height(point["total"] * (1.0 - point["score"])):.1f}" fill="#e76f51" opacity="0.58"/>'
        for point in candidate_points
    )
    x_ticks = "\n".join(
        f'  <text x="{x_coord(iteration):.1f}" y="{fail_y + fail_h + 22}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#555">{iteration}</text>'
        for iteration in sorted(set(iterations))
    )
    score_ticks = [score_y_min, (score_y_min + score_y_max) / 2, score_y_max]
    score_tick_lines = "\n".join(
        f'  <line x1="{chart_x}" y1="{y_coord(score):.1f}" x2="{chart_x + chart_w}" y2="{y_coord(score):.1f}" stroke="#ddd" stroke-dasharray="3 4"/>'
        for score in score_ticks
    )
    score_tick_labels = "\n".join(
        f'  <text x="{chart_x - 10}" y="{y_coord(score) + 4:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="11" fill="#555">{score:.0%}</text>'
        for score in score_ticks
    )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="920" height="520" viewBox="0 0 920 520">
  <rect width="920" height="520" fill="#ffffff"/>
  <text x="40" y="48" font-family="Arial, sans-serif" font-size="24" font-weight="700" fill="#222">AIME {optimizer_label} Run {Path(summary['run_dir']).name}</text>
  <text x="40" y="82" font-family="Arial, sans-serif" font-size="15" fill="#555">Best validation: {best_val:.2%} | Pareto aggregate: {pareto:.2%} | Candidates: {summary['num_candidates']}</text>
  <text x="58" y="108" font-family="Arial, sans-serif" font-size="17" font-weight="700" fill="#222">Optimization progress</text>
  <rect x="{chart_x}" y="{chart_y}" width="{chart_w}" height="{chart_h}" fill="#fafafa" stroke="#ddd"/>
{score_tick_lines}
{score_tick_labels}
{best_polyline}
{pareto_polyline}
{candidate_circles}
{pareto_dots}
  <circle cx="70" cy="328" r="4.8" fill="#355070"/><text x="82" y="332" font-family="Arial, sans-serif" font-size="12" fill="#333">candidate</text>
  <line x1="160" y1="328" x2="190" y2="328" stroke="#2a9d8f" stroke-width="3"/><text x="198" y="332" font-family="Arial, sans-serif" font-size="12" fill="#333">best so far</text>
  <line x1="295" y1="328" x2="325" y2="328" stroke="#f4a261" stroke-width="3" stroke-dasharray="5 5"/><text x="333" y="332" font-family="Arial, sans-serif" font-size="12" fill="#333">pareto</text>
  <text x="58" y="350" font-family="Arial, sans-serif" font-size="15" font-weight="700" fill="#222">Failed validation examples</text>
  <rect x="{chart_x}" y="{fail_y}" width="{chart_w}" height="{fail_h}" fill="#fafafa" stroke="#ddd"/>
{fail_bars}
  {x_ticks}
  <text x="{chart_x + chart_w / 2}" y="510" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#555">iteration</text>
  <text x="540" y="126" font-family="Arial, sans-serif" font-size="20" font-weight="700" fill="#222">Final AIME Test pass@{test_results['pass_k']}</text>
  <line x1="540" y1="430" x2="850" y2="430" stroke="#bbb"/>
  <rect x="585" y="{430 - baseline_height:.1f}" width="80" height="{baseline_height:.1f}" fill="#6c757d"/>
  <rect x="725" y="{430 - optimized_height:.1f}" width="80" height="{optimized_height:.1f}" fill="#2a9d8f"/>
  <text x="625" y="{410 - baseline_height:.1f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="17" font-weight="700" fill="#222">{baseline:.2%}</text>
  <text x="765" y="{410 - optimized_height:.1f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="17" font-weight="700" fill="#222">{optimized:.2%}</text>
  <text x="625" y="462" text-anchor="middle" font-family="Arial, sans-serif" font-size="15" fill="#333">Baseline</text>
  <text x="765" y="462" text-anchor="middle" font-family="Arial, sans-serif" font-size="15" fill="#333">Optimized</text>
  <text x="695" y="494" text-anchor="middle" font-family="Arial, sans-serif" font-size="18" font-weight="700" fill="#2a9d8f">pass improvement: +{improvement:.2%}</text>
  {mean_line}
</svg>
"""
    output_path.write_text(svg, encoding="utf-8")
