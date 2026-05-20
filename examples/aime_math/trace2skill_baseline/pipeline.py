from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from examples.aime_math.trace2skill_baseline.config import Trace2SkillBaselineConfig
from examples.aime_math.trace2skill_baseline.stage1_rollout import TrajectoryCollector
from examples.aime_math.trace2skill_baseline.stage2_analysts import ParallelAnalystRunner
from examples.aime_math.trace2skill_baseline.stage3_merge import SuggestionMerger
from examples.aime_math.trace2skill_baseline.types import Trace2SkillBaselineResult
from examples.aime_math.trace2skill_baseline.utils import write_json


class Trace2SkillBaselineRunner:
    def __init__(self, experiment: Any, baseline_config: Trace2SkillBaselineConfig):
        self.experiment = experiment
        self.baseline_config = baseline_config
        self.output_dir = Path(experiment.config.run_dir) / "trace2skill_baseline"

    def run(self) -> Trace2SkillBaselineResult:
        initial_prompt = self.experiment.config.initial_prompt
        trainset_size = len(self.experiment.trainset)
        if self.experiment.config.max_metric_calls <= 0:
            raise ValueError(
                "Trace2Skill baseline received a non-positive rollout budget. "
                "Set AIME_MAX_METRIC_CALLS to a positive integer."
            )

        print(
            "[AIME][Trace2Skill] Running baseline with "
            f"mode={self.baseline_config.mode}, trainset_size={trainset_size}, "
            f"budget_cap={self.experiment.config.max_metric_calls}, "
            f"merge_fanin(B)={self.baseline_config.merge_fanin}, "
            f"error_analyst_max_turns={self.baseline_config.error_analyst_max_turns}"
        )

        collector = TrajectoryCollector(self.experiment, self.output_dir)
        analyst_runner = ParallelAnalystRunner(
            self.experiment,
            self.output_dir,
            error_analyst_max_turns=self.baseline_config.error_analyst_max_turns,
        )
        merger = SuggestionMerger(self.experiment.reflection_lm, self.output_dir)

        rng = random.Random(self.experiment.config.seed)
        remaining_budget = self.experiment.config.max_metric_calls
        valset_metric_cost = len(self.experiment.valset)
        current_prompt = initial_prompt
        final_val_score: float | None = None
        final_val_mean: float | None = None
        total_metric_calls = 0
        total_trajectories = 0
        total_suggestions = 0
        iteration_idx = 0
        iterations: list[dict[str, object]] = []

        while remaining_budget > 0:
            estimated_iteration_cost = trainset_size + valset_metric_cost
            if remaining_budget < estimated_iteration_cost:
                print(
                    "[AIME][Trace2Skill] Stopping optimization before next iteration: "
                    f"remaining_budget={remaining_budget} is smaller than full iteration cost="
                    f"{estimated_iteration_cost} (rollout_size={trainset_size}, "
                    f"valset_size={valset_metric_cost}). "
                    "Proceeding to final evaluation with the best prompt so far."
                )
                break

            iteration_idx += 1
            rollout_size = trainset_size
            shuffled_batch = list(self.experiment.trainset)
            rng.shuffle(shuffled_batch)
            batch = shuffled_batch[:rollout_size]

            trajectories = collector.collect(current_prompt, batch, rollout_size, iteration_idx)
            suggestions = analyst_runner.analyze(
                current_prompt,
                trajectories,
                mode=self.baseline_config.mode,
                iteration_idx=iteration_idx,
            )
            next_prompt, merge_history = merger.merge_into_prompt(
                current_prompt,
                suggestions,
                merge_fanin=self.baseline_config.merge_fanin,
                iteration_idx=iteration_idx,
            )

            val_stats = self.experiment.evaluate_prompt_summary(
                next_prompt,
                self.experiment.valset,
                pass_k=1,
                cache_label=f"trace2skill_baseline_val_iter_{iteration_idx:03d}",
            )

            rollout_metric_calls = len(trajectories)
            val_metric_calls = int(val_stats["total_attempts"])
            iteration_metric_calls = rollout_metric_calls + val_metric_calls

            remaining_budget -= iteration_metric_calls
            total_metric_calls += iteration_metric_calls
            total_trajectories += rollout_metric_calls
            total_suggestions += len(suggestions)
            current_prompt = next_prompt
            iterations.append(
                {
                    "iteration": iteration_idx,
                    "rollout_size": rollout_metric_calls,
                    "rollout_metric_calls": rollout_metric_calls,
                    "val_metric_calls": val_metric_calls,
                    "iteration_metric_calls": iteration_metric_calls,
                    "remaining_budget": remaining_budget,
                    "num_suggestions": len(suggestions),
                    "val_score": val_stats["pass_score"],
                    "val_mean_score": val_stats["mean_score"],
                    "merge_rounds": len(merge_history),
                }
            )

            print(
                "[AIME][Trace2Skill] Iteration complete: "
                f"iteration={iteration_idx}, rollout_size={rollout_metric_calls}, "
                f"val_metric_calls={val_metric_calls}, val_score={val_stats['pass_score']:.2%}, "
                f"remaining_budget={remaining_budget}"
            )
            final_val_score = float(val_stats["pass_score"])
            final_val_mean = float(val_stats["mean_score"])

        prompt_path = self.output_dir / "optimized_prompt.txt"
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(current_prompt, encoding="utf-8")
        write_json(
            self.output_dir / "summary.json",
            {
                "initial_prompt": initial_prompt,
                "optimized_prompt": current_prompt,
                "mode": self.baseline_config.mode,
                "trainset_size": trainset_size,
                "valset_size": valset_metric_cost,
                "budget_cap": self.experiment.config.max_metric_calls,
                "metric_calls_used": total_metric_calls,
                "num_trajectories": total_trajectories,
                "rollout_metric_calls": total_trajectories,
                "val_metric_calls": total_metric_calls - total_trajectories,
                "num_suggestions": total_suggestions,
                "num_iterations": iteration_idx,
                "final_val_score": final_val_score,
                "final_val_mean_score": final_val_mean,
                "iterations": iterations,
            },
        )

        return Trace2SkillBaselineResult(
            optimized_prompt=current_prompt,
            val_score=float(final_val_score) if final_val_score is not None else float("-inf"),
            metric_calls_used=total_metric_calls,
            num_trajectories=total_trajectories,
            num_suggestions=total_suggestions,
            num_iterations=iteration_idx,
            output_dir=self.output_dir,
        )
