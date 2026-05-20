from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from examples.aime_math.trace2skill_baseline.types import AnalystSuggestion, TrajectoryRecord
from examples.aime_math.trace2skill_baseline.utils import extract_tagged_block, write_jsonl


def _trajectory_digest(trajectory: TrajectoryRecord) -> str:
    return (
        f"Question:\n{trajectory.input_text}\n\n"
        f"Gold answer: {trajectory.gold_answer}\n"
        f"Model answer: {trajectory.model_answer}\n"
        f"Score: {trajectory.score:.1f}\n"
        f"Reasoning trace:\n{trajectory.reasoning or '(none)'}\n\n"
        f"Feedback:\n{trajectory.feedback}"
    )


class SuccessAnalyst:
    def __init__(self, lm):
        self.lm = lm

    def analyze(self, prompt: str, trajectory: TrajectoryRecord) -> str | None:
        response = self.lm(
            f"""You are a success analyst in a Trace2Skill-style prompt distillation pipeline.

Current prompt:
<prompt>
{prompt}
</prompt>

Analyze this SUCCESSFUL trajectory and extract one concise, generalizable improvement suggestion for the prompt.
The suggestion must be applicable to future AIME-style math problems, not just this single example.
If there is no useful generalizable lesson, return NONE.

Trajectory:
{_trajectory_digest(trajectory)}

Return only:
<suggestion>...</suggestion>
"""
        )
        suggestion = extract_tagged_block(response, "suggestion") or response.strip()
        if suggestion.strip().upper() == "NONE":
            return None
        return suggestion.strip()


class ErrorAnalyst:
    def __init__(self, lm, max_turns: int):
        self.lm = lm
        self.max_turns = max_turns

    def analyze(self, prompt: str, trajectory: TrajectoryRecord) -> str | None:
        diagnosis_response = self.lm(
            f"""You are an error analyst in a Trace2Skill-style prompt distillation pipeline.

Current prompt:
<prompt>
{prompt}
</prompt>

Analyze this FAILED trajectory. Identify the most likely root cause in the current prompt, using only evidence from
the trajectory and feedback. Focus on a lesson that could improve future AIME-style math problems.

Trajectory:
{_trajectory_digest(trajectory)}

Return only:
<analysis>...</analysis>
"""
        )
        diagnosis = extract_tagged_block(diagnosis_response, "analysis") or diagnosis_response.strip()

        for _ in range(max(0, self.max_turns - 2)):
            refinement_response = self.lm(
                f"""Refine the following failure diagnosis so it stays causal, concise, and generalizable.

Current diagnosis:
<analysis>
{diagnosis}
</analysis>

Trajectory:
{_trajectory_digest(trajectory)}

Return only:
<analysis>...</analysis>
"""
            )
            refined = extract_tagged_block(refinement_response, "analysis") or refinement_response.strip()
            if refined:
                diagnosis = refined

        suggestion_response = self.lm(
            f"""You are finishing an error-analysis pass in a Trace2Skill-style prompt distillation pipeline.

Current prompt:
<prompt>
{prompt}
</prompt>

Failure diagnosis:
<analysis>
{diagnosis}
</analysis>

Turn the diagnosis into one concise, direct improvement suggestion for the prompt.
The suggestion must be generalizable beyond this example. If no safe generalizable lesson exists, return NONE.

Trajectory:
{_trajectory_digest(trajectory)}

Return only:
<suggestion>...</suggestion>
"""
        )
        suggestion = extract_tagged_block(suggestion_response, "suggestion") or suggestion_response.strip()
        if suggestion.strip().upper() == "NONE":
            return None
        return suggestion.strip()


class ParallelAnalystRunner:
    def __init__(self, experiment: Any, output_dir: Path, error_analyst_max_turns: int):
        self.experiment = experiment
        self.output_dir = output_dir
        self.success_analyst = SuccessAnalyst(experiment.reflection_lm)
        self.error_analyst = ErrorAnalyst(experiment.reflection_lm, max_turns=error_analyst_max_turns)

    def analyze(
        self,
        prompt: str,
        trajectories: list[TrajectoryRecord],
        mode: str,
        iteration_idx: int,
    ) -> list[AnalystSuggestion]:
        candidates: list[tuple[str, TrajectoryRecord]] = []
        for trajectory in trajectories:
            if trajectory.is_success and mode in {"success", "combined"}:
                candidates.append(("success", trajectory))
            elif not trajectory.is_success and mode in {"error", "combined"}:
                candidates.append(("error", trajectory))

        print(
            "[AIME][Trace2Skill] Stage 2 analysts: "
            f"iteration={iteration_idx}, processing {len(candidates)} trajectories in mode={mode!r}."
        )

        suggestions: list[AnalystSuggestion] = []
        max_workers = min(max(1, self.experiment.config.max_workers), max(1, len(candidates)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_item = {}
            for analyst_kind, trajectory in candidates:
                if analyst_kind == "success":
                    future = executor.submit(self.success_analyst.analyze, prompt, trajectory)
                else:
                    future = executor.submit(self.error_analyst.analyze, prompt, trajectory)
                future_to_item[future] = (analyst_kind, trajectory)

            for future in as_completed(future_to_item):
                analyst_kind, trajectory = future_to_item[future]
                suggestion = future.result()
                if suggestion:
                    suggestions.append(
                        AnalystSuggestion(
                            trajectory_id=trajectory.trajectory_id,
                            analyst_kind=analyst_kind,
                            source_score=trajectory.score,
                            suggestion=suggestion,
                        )
                    )

        suggestions.sort(key=lambda item: (item.analyst_kind, item.trajectory_id))
        write_jsonl(
            self.output_dir / f"stage2_suggestions_iter_{iteration_idx:03d}.jsonl",
            [item.to_dict() for item in suggestions],
        )
        return suggestions
