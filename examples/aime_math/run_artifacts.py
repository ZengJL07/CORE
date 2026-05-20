from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

DEFAULT_VALSET_TOTAL = 45


@contextmanager
def capture_run_logs(run_dir: Path) -> Iterator[None]:
    run_dir.mkdir(parents=True, exist_ok=True)
    with _build_logger(run_dir / "run_log.txt"):
        yield


def infer_valset_total_from_summary(payload: dict[str, Any], default: int = DEFAULT_VALSET_TOTAL) -> int:
    valset_size = payload.get("valset_size")
    if isinstance(valset_size, int) and valset_size > 0:
        return valset_size

    for candidate in payload.get("candidate_pool", []):
        if not isinstance(candidate, dict):
            continue
        val_scores = candidate.get("val_scores_by_sample")
        if isinstance(val_scores, dict) and val_scores:
            return len(val_scores)

    return default


def build_best_events(candidate_points: list[dict[str, Any]]) -> list[dict[str, float]]:
    best_events: list[dict[str, float]] = []
    best_score: float | None = None

    for point in candidate_points:
        score = float(point["score"])
        if best_score is None or score > best_score:
            best_score = score
            best_events.append({"iteration": int(point["iteration"]), "score": score})

    return best_events


def load_sample_weighted_gepa_candidate_points(payload: dict[str, Any]) -> list[dict[str, Any]]:
    candidate_points = payload.get("candidate_points")
    if isinstance(candidate_points, list) and candidate_points:
        return [_normalize_candidate_point(point, fallback_iteration=idx) for idx, point in enumerate(candidate_points, start=1)]

    return reconstruct_sample_weighted_gepa_candidate_points(payload)


def reconstruct_sample_weighted_gepa_candidate_points(payload: dict[str, Any]) -> list[dict[str, Any]]:
    outer_step_records = payload.get("outer_step_records")
    if not isinstance(outer_step_records, list) or not outer_step_records:
        return []

    initial_prompt = _infer_initial_prompt(outer_step_records)
    candidate_prompts_by_id: dict[int, str] = {}
    current_pool_by_prompt: dict[str, int] = {}

    if initial_prompt is not None:
        candidate_prompts_by_id[0] = initial_prompt
        current_pool_by_prompt[initial_prompt] = 0

    candidate_points: list[dict[str, Any]] = []
    next_iteration = 1

    for outer_step_record in outer_step_records:
        branch_records = outer_step_record.get("branch_records", [])
        for branch in branch_records:
            current_candidate_id = branch.get("current_candidate_id")
            current_prompt = branch.get("current_prompt")
            if isinstance(current_candidate_id, int) and isinstance(current_prompt, str) and current_prompt.strip():
                candidate_prompts_by_id[current_candidate_id] = current_prompt

        surviving_by_prompt: dict[str, dict[str, Any]] = {}
        for branch in branch_records:
            if not branch.get("passed_total_gate"):
                continue
            final_prompt = branch.get("final_prompt")
            if not isinstance(final_prompt, str) or not final_prompt.strip():
                continue
            existing = surviving_by_prompt.get(final_prompt)
            if existing is None or _branch_total_score(branch) > _branch_total_score(existing):
                surviving_by_prompt[final_prompt] = branch

        inserted_candidate_ids = iter(
            candidate_id
            for candidate_id in outer_step_record.get("inserted_candidate_ids", [])
            if isinstance(candidate_id, int)
        )

        for final_prompt, branch in surviving_by_prompt.items():
            if final_prompt in current_pool_by_prompt:
                continue

            candidate_id = next(inserted_candidate_ids, None)
            if candidate_id is None:
                break

            full_val_score = branch.get("full_val_score")
            if full_val_score is None:
                continue

            candidate_points.append(
                {
                    "iteration": next_iteration,
                    "outer_step": int(outer_step_record.get("outer_step", 0)),
                    "branch_idx": int(branch.get("branch_idx", 0)),
                    "candidate_id": candidate_id,
                    "score": float(full_val_score),
                    "accepted_to_pool": bool(branch.get("accepted_to_pool", False)),
                }
            )
            candidate_prompts_by_id[candidate_id] = final_prompt
            next_iteration += 1

        pool_candidate_ids = outer_step_record.get("metadata", {}).get("pool_candidate_ids", [])
        next_pool_by_prompt: dict[str, int] = {}
        for candidate_id in pool_candidate_ids:
            if not isinstance(candidate_id, int):
                continue
            prompt = candidate_prompts_by_id.get(candidate_id)
            if prompt is not None:
                next_pool_by_prompt[prompt] = candidate_id
        if next_pool_by_prompt:
            current_pool_by_prompt = next_pool_by_prompt

    return candidate_points


def _normalize_candidate_point(point: Any, *, fallback_iteration: int) -> dict[str, Any]:
    if not isinstance(point, dict):
        raise ValueError(f"Candidate point must be a dict, got {type(point)}")
    return {
        "iteration": int(point.get("iteration", fallback_iteration)),
        "outer_step": int(point.get("outer_step", 0)),
        "branch_idx": int(point.get("branch_idx", 0)),
        "candidate_id": int(point.get("candidate_id", -1)),
        "score": float(point["score"]),
        "accepted_to_pool": bool(point.get("accepted_to_pool", False)),
    }


def _infer_initial_prompt(outer_step_records: list[dict[str, Any]]) -> str | None:
    for outer_step_record in outer_step_records:
        branch_records = outer_step_record.get("branch_records", [])
        if not branch_records:
            continue
        current_prompt = branch_records[0].get("current_prompt")
        if isinstance(current_prompt, str) and current_prompt.strip():
            return current_prompt
    return None


def _branch_total_score(branch: dict[str, Any]) -> float:
    total_score = branch.get("total_score")
    if total_score is None:
        return float("-inf")
    return float(total_score)


def _build_logger(log_path: Path):
    try:
        from gepa.logging.logger import Logger

        return Logger(str(log_path), mode="w")
    except ModuleNotFoundError:
        return _FallbackLogger(log_path)


class _FallbackTee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj: str) -> None:
        for file_obj in self.files:
            file_obj.write(obj)

    def flush(self) -> None:
        for file_obj in self.files:
            if hasattr(file_obj, "flush"):
                file_obj.flush()


class _FallbackLogger:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.stderr_path = Path(str(log_path).replace("run_log.", "run_log_stderr."))

    def __enter__(self):
        import sys

        self._stdout = sys.stdout
        self._stderr = sys.stderr
        self._stdout_handle = self.log_path.open("w", encoding="utf-8")
        self._stderr_handle = self.stderr_path.open("w", encoding="utf-8")
        sys.stdout = _FallbackTee(sys.stdout, self._stdout_handle)
        sys.stderr = _FallbackTee(sys.stderr, self._stderr_handle)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        import sys

        sys.stdout = self._stdout
        sys.stderr = self._stderr
        self._stdout_handle.close()
        self._stderr_handle.close()
