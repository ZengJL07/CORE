from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from examples.aime_math.experiment import AIMEExperiment
from examples.aime_math.prompt_optimizers.autograd import build_textual_batch_loss
from examples.aime_math.prompt_optimizers.optimizer import (
    TextGradAlgorithmConfig,
    build_prompt_optimizer,
)
from examples.aime_math.prompt_optimizers.state import PromptHistoryEntry, PromptVariable


@dataclass(frozen=True)
class TextGradRunResult:
    best_prompt: str
    best_val_score: float
    prompt_history: list[PromptHistoryEntry]
    total_metric_calls: int
    total_steps: int


class TextGradRunner:
    def __init__(self, experiment: AIMEExperiment, config: TextGradAlgorithmConfig):
        self.experiment = experiment
        self.config = config
        self.rng = random.Random(experiment.config.seed)
        self.variable = PromptVariable(
            value=experiment.config.initial_prompt,
            role_description=(
                "system prompt for solving AIME-style math problems with strict integer-only final answers"
            ),
        )
        self.optimizer = build_prompt_optimizer(
            variable=self.variable,
            engine=experiment.reflection_lm,
            config=config,
            rng=self.rng,
        )

    def _iter_train_batches(self):
        trainset = list(self.experiment.trainset)
        for _epoch in range(self.config.max_epochs):
            indices = list(range(len(trainset)))
            self.rng.shuffle(indices)
            for start in range(0, len(indices), self.config.batch_size):
                batch_indices = indices[start : start + self.config.batch_size]
                yield [trainset[idx] for idx in batch_indices]

    def run(self) -> TextGradRunResult:
        print(
            "[AIME] TextGrad config: "
            f"algorithm={self.config.algorithm}, batch_size={self.config.batch_size}, "
            f"max_epochs={self.config.max_epochs}, max_steps={self.config.max_steps}, "
            f"candidate_pool_size={self.config.candidate_pool_size}, top_k={self.config.top_k}, "
            f"gumbel_tau={self.config.gumbel_temperature}, momentum_buffer_size={self.config.momentum_buffer_size}, "
            f"momentum_size={self.config.momentum_size}, bootstrap_samples={self.config.bootstrap_samples}, "
            f"validation_frequency={self.config.validation_frequency}, "
            f"revert_on_validation_drop={self.config.revert_on_validation_drop}"
        )

        prompt_history: list[PromptHistoryEntry] = []
        total_metric_calls = 0
        best_prompt = self.variable.value
        best_val_stats = self.experiment.evaluate_prompt_summary(
            best_prompt,
            self.experiment.valset,
            pass_k=1,
            cache_label=f"{self.experiment.config.backend}_{self.config.algorithm}_initial_val",
        )
        best_val_score = float(best_val_stats["mean_score"])
        total_metric_calls = int(best_val_stats["total_attempts"])
        previous_prompt = best_prompt
        previous_val_score = best_val_score
        print(f"[AIME] Initial validation score: {best_val_score:.2%}")
        valset_metric_cost = len(self.experiment.valset)

        step_idx = 0
        for batch in self._iter_train_batches():
            remaining_budget = self.experiment.config.max_metric_calls - total_metric_calls
            estimated_step_cost = self.optimizer.estimate_metric_calls(len(batch))
            if step_idx >= self.config.max_steps or remaining_budget <= 0:
                break
            if estimated_step_cost > remaining_budget:
                print(
                    f"[AIME] Stopping before step {step_idx + 1}: remaining budget={remaining_budget}, "
                    f"estimated_step_cost={estimated_step_cost}"
                )
                break

            print(
                f"[AIME] TextGrad step {step_idx + 1}: "
                f"batch_size={len(batch)}, current_prompt_chars={len(self.variable.value)}"
            )
            self.optimizer.zero_grad()
            current_batch_evaluation = self.experiment.evaluate_prompt_on_batch(self.variable.value, batch)
            batch_loss = build_textual_batch_loss(current_batch_evaluation, self.variable)
            batch_loss.backward(self.experiment.reflection_lm)
            step_result = self.optimizer.step(
                experiment=self.experiment,
                batch=batch,
                current_batch_evaluation=current_batch_evaluation,
                step_idx=step_idx,
                prompt_history=prompt_history,
            )
            total_metric_calls += step_result.metric_calls

            val_score = None
            if step_idx % self.config.validation_frequency == 0:
                remaining_budget = self.experiment.config.max_metric_calls - total_metric_calls
                if remaining_budget >= valset_metric_cost:
                    val_stats = self.experiment.evaluate_prompt_summary(
                        step_result.prompt,
                        self.experiment.valset,
                        pass_k=1,
                        cache_label=f"{self.experiment.config.backend}_{self.config.algorithm}_val_step_{step_idx}",
                    )
                    total_metric_calls += int(val_stats["total_attempts"])
                    val_score = float(val_stats["mean_score"])
                    if self.config.revert_on_validation_drop and val_score < previous_val_score:
                        print(
                            f"[AIME] Validation dropped from {previous_val_score:.2%} to {val_score:.2%}; "
                            "reverting to previous prompt."
                        )
                        self.variable.set_value(previous_prompt)
                        val_score = previous_val_score
                    else:
                        previous_prompt = step_result.prompt
                        previous_val_score = val_score

                    if val_score > best_val_score:
                        best_val_score = val_score
                        best_prompt = self.variable.value
                        print(f"[AIME] New best TextGrad prompt on valset: {best_val_score:.2%}")
                else:
                    print(
                        f"[AIME] Skipping validation at step {step_idx + 1}: "
                        f"remaining budget={remaining_budget}, required={valset_metric_cost}"
                    )

            prompt_history.append(
                PromptHistoryEntry(
                    step_idx=step_idx,
                    prompt=self.variable.value,
                    batch_score=step_result.updated_batch_score,
                    val_score=val_score,
                    metadata={
                        "current_batch_score": step_result.current_batch_score,
                        "combined_gradient": step_result.combined_gradient,
                        **step_result.metadata,
                    },
                )
            )
            print(
                f"[AIME] Step {step_idx + 1} finished: "
                f"batch_score={step_result.updated_batch_score:.2%}, "
                f"val_score={'n/a' if val_score is None else f'{val_score:.2%}'}, "
                f"metric_calls={total_metric_calls}"
            )
            step_idx += 1

        self._write_summary(prompt_history, best_prompt, best_val_score, total_metric_calls, step_idx)
        return TextGradRunResult(
            best_prompt=best_prompt,
            best_val_score=best_val_score,
            prompt_history=prompt_history,
            total_metric_calls=total_metric_calls,
            total_steps=step_idx,
        )

    def _write_summary(
        self,
        prompt_history: list[PromptHistoryEntry],
        best_prompt: str,
        best_val_score: float,
        total_metric_calls: int,
        total_steps: int,
    ) -> None:
        summary = {
            "algorithm": self.config.algorithm,
            "best_prompt": best_prompt,
            "best_val_score": best_val_score,
            "total_metric_calls": total_metric_calls,
            "total_steps": total_steps,
            "prompt_history": [
                {
                    "step_idx": entry.step_idx,
                    "prompt": entry.prompt,
                    "batch_score": entry.batch_score,
                    "val_score": entry.val_score,
                    "metadata": entry.metadata,
                }
                for entry in prompt_history
            ],
        }
        Path(self.experiment.config.run_dir).mkdir(parents=True, exist_ok=True)
        summary_path = Path(self.experiment.config.run_dir) / f"textgrad_{self.config.algorithm}_summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
