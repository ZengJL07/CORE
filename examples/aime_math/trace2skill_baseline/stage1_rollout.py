from __future__ import annotations

from pathlib import Path
from typing import Any

from examples.aime_math.trace2skill_baseline.types import TrajectoryRecord
from examples.aime_math.trace2skill_baseline.utils import write_jsonl


class TrajectoryCollector:
    def __init__(self, experiment: Any, output_dir: Path):
        self.experiment = experiment
        self.output_dir = output_dir

    def collect(self, prompt: str, dataset: list[Any], budget: int, iteration_idx: int) -> list[TrajectoryRecord]:
        batch = list(dataset[:budget])
        print(
            "[AIME][Trace2Skill] Stage 1 rollout: "
            f"iteration={iteration_idx}, collecting {len(batch)} trajectories with one sample per selected training problem."
        )
        batch_eval = self.experiment.evaluate_prompt_on_batch(prompt, batch)

        trajectories = []
        for idx, item in enumerate(batch_eval.example_evaluations):
            side_info = item.side_info
            if side_info.get("problem_id") is not None:
                failed_details = side_info.get("failed_details", []) or []
                first_failure = failed_details[0] if failed_details else {}
                feedback = str(side_info.get("execution_feedback", ""))
                if first_failure.get("test"):
                    feedback += f"\nFirst failing test: {first_failure.get('test', '')}"
                if first_failure.get("error"):
                    feedback += f"\nFailure detail: {first_failure.get('error', '')}"
                if side_info.get("traceback"):
                    feedback += f"\nTraceback:\n{side_info.get('traceback', '')}"

                trajectories.append(
                    TrajectoryRecord(
                        trajectory_id=f"iter_{iteration_idx:03d}_traj_{idx:03d}",
                        input_text=str(side_info.get("input", getattr(item.example, "input", ""))),
                        gold_answer=str(getattr(item.example, "canonical_solution", getattr(item.example, "answer", ""))),
                        model_answer=str(side_info.get("output", "")),
                        reasoning=str(side_info.get("stdout", "")),
                        score=float(item.score),
                        feedback=feedback,
                    )
                )
                continue
            trajectories.append(
                TrajectoryRecord(
                    trajectory_id=f"iter_{iteration_idx:03d}_traj_{idx:03d}",
                    input_text=str(side_info.get("input", getattr(item.example, "input", ""))),
                    gold_answer=str(getattr(item.example, "answer", "")),
                    model_answer=str(side_info.get("output", "")),
                    reasoning=str(side_info.get("reasoning", "")),
                    score=float(item.score),
                    feedback=str(side_info.get("execution_feedback", "")),
                )
            )

        write_jsonl(
            self.output_dir / f"stage1_trajectories_iter_{iteration_idx:03d}.jsonl",
            [item.to_dict() for item in trajectories],
        )
        return trajectories
