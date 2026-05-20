from __future__ import annotations

import json
import os
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from examples.aime_math.config import _env_bool, _env_float, _env_int
from examples.aime_math.prompt_optimizers.autograd import extract_tagged_text
from examples.aime_math.prompt_optimizers.sampling import (
    gumbel_topk_indices,
    softmax,
    weighted_sample_without_replacement,
)
from examples.aime_math.prompt_optimizers.state import (
    OptimizationStepArtifacts,
    PromptHistoryEntry,
    PromptVariable,
)

if TYPE_CHECKING:
    from examples.aime_math.experiment import AIMEExperiment, BatchEvaluation


PROMPT_TAG = "IMPROVED_PROMPT"
MOMENTUM_TAG = "MOMENTUM_SUMMARY"


@dataclass(frozen=True)
class TextGradAlgorithmConfig:
    algorithm: str
    batch_size: int
    max_epochs: int
    max_steps: int
    candidate_pool_size: int
    top_k: int
    gumbel_temperature: float
    momentum_buffer_size: int
    momentum_size: int
    bootstrap_samples: int
    validation_frequency: int
    revert_on_validation_drop: bool

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.max_epochs < 1:
            raise ValueError(f"max_epochs must be >= 1, got {self.max_epochs}")
        if self.max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {self.max_steps}")
        if self.validation_frequency < 1:
            raise ValueError(f"validation_frequency must be >= 1, got {self.validation_frequency}")
        if self.candidate_pool_size < 1:
            raise ValueError(f"candidate_pool_size must be >= 1, got {self.candidate_pool_size}")
        if self.top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {self.top_k}")
        if self.bootstrap_samples < 1:
            raise ValueError(f"bootstrap_samples must be >= 1, got {self.bootstrap_samples}")

    @classmethod
    def from_env(cls) -> "TextGradAlgorithmConfig":
        return cls(
            algorithm=os.environ.get("AIME_TEXTGRAD_ALGORITHM", "tgd").strip().lower(),
            batch_size=_env_int("AIME_TEXTGRAD_BATCH_SIZE", 3),
            max_epochs=_env_int("AIME_TEXTGRAD_MAX_EPOCHS", 3),
            max_steps=_env_int("AIME_TEXTGRAD_MAX_STEPS", 64),
            candidate_pool_size=_env_int("AIME_TEXTGRAD_CANDIDATE_POOL_SIZE", 4),
            top_k=_env_int("AIME_TEXTGRAD_TOP_K", 2),
            gumbel_temperature=_env_float("AIME_TEXTGRAD_GUMBEL_TAU", 1.0),
            momentum_buffer_size=_env_int("AIME_TEXTGRAD_MOMENTUM_BUFFER_SIZE", 4),
            momentum_size=_env_int("AIME_TEXTGRAD_MOMENTUM_SIZE", 2),
            bootstrap_samples=_env_int("AIME_TEXTGRAD_BOOTSTRAP_SAMPLES", 4),
            validation_frequency=_env_int("AIME_TEXTGRAD_VALIDATION_FREQUENCY", 1),
            revert_on_validation_drop=_env_bool("AIME_TEXTGRAD_REVERT_ON_VALIDATION_DROP", False),
        )


class BasePromptOptimizer(ABC):
    def __init__(
        self,
        *,
        variable: PromptVariable,
        engine,
        config: TextGradAlgorithmConfig,
        rng: random.Random,
    ) -> None:
        self.variable = variable
        self.engine = engine
        self.config = config
        self.rng = rng

    def zero_grad(self) -> None:
        self.variable.reset_gradients()

    def _gradient_signal(self) -> str:
        return self.variable.get_gradient_and_context_text() or self.variable.get_gradient_text()

    def _update_prompt(self, gradient_signal: str, *, extra_context: str = "") -> str:
        prompt = "\n".join(
            [
                "You are updating a math-solving system prompt.",
                "Revise the prompt using the feedback below. Keep the output as a drop-in replacement prompt.",
                f"Role of the variable: {self.variable.get_role_description()}",
                f"Current prompt:\n{self.variable.get_short_value()}",
                f"Context and feedback:\n{gradient_signal}",
                extra_context.strip(),
                f"Return only the improved prompt between <{PROMPT_TAG}> and </{PROMPT_TAG}>.",
            ]
        ).strip()
        response = self.engine(prompt)
        return extract_tagged_text(response, PROMPT_TAG)

    def _score_prompt(self, experiment: "AIMEExperiment", prompt: str, batch: list[Any]) -> tuple[float, int]:
        batch_eval = experiment.evaluate_prompt_on_batch(prompt, batch)
        return batch_eval.average_score, batch_eval.metric_calls

    def _bootstrap_score(
        self,
        experiment: "AIMEExperiment",
        prompt: str,
        batch: list[Any],
    ) -> tuple[float, int]:
        if not batch:
            return 0.0, 0

        total_score = 0.0
        total_metric_calls = 0
        for _ in range(self.config.bootstrap_samples):
            sampled_batch = [batch[self.rng.randrange(len(batch))] for _ in range(len(batch))]
            score, metric_calls = self._score_prompt(experiment, prompt, sampled_batch)
            total_score += score
            total_metric_calls += metric_calls
        return total_score / self.config.bootstrap_samples, total_metric_calls

    def _generate_candidate_pool(self, combined_gradient: str) -> list[str]:
        candidates: list[str] = []
        for idx in range(self.config.candidate_pool_size):
            updated_prompt = self._update_prompt(
                combined_gradient,
                extra_context=(
                    f"Candidate variation #{idx + 1}: explore a distinct revision strategy while staying faithful "
                    "to the task and strict answer-format constraints."
                ),
            )
            if updated_prompt not in candidates:
                candidates.append(updated_prompt)
        return candidates or [self.variable.value]

    @abstractmethod
    def estimate_metric_calls(self, batch_size: int) -> int:
        raise NotImplementedError

    @abstractmethod
    def step(
        self,
        *,
        experiment: "AIMEExperiment",
        batch: list[Any],
        current_batch_evaluation: "BatchEvaluation",
        step_idx: int,
        prompt_history: list[PromptHistoryEntry],
    ) -> OptimizationStepArtifacts:
        raise NotImplementedError


class TextualGradientDescentOptimizer(BasePromptOptimizer):
    def estimate_metric_calls(self, batch_size: int) -> int:
        return batch_size * 2

    def step(
        self,
        *,
        experiment: "AIMEExperiment",
        batch: list[Any],
        current_batch_evaluation: "BatchEvaluation",
        step_idx: int,
        prompt_history: list[PromptHistoryEntry],
    ) -> OptimizationStepArtifacts:
        gradient_signal = self._gradient_signal()
        new_prompt = self._update_prompt(gradient_signal)
        updated_score, updated_metric_calls = self._score_prompt(experiment, new_prompt, batch)
        self.variable.set_value(new_prompt)

        return OptimizationStepArtifacts(
            prompt=new_prompt,
            current_batch_score=current_batch_evaluation.average_score,
            updated_batch_score=updated_score,
            combined_gradient=gradient_signal,
            metric_calls=current_batch_evaluation.metric_calls + updated_metric_calls,
            metadata={"algorithm": "tgd", "step_idx": step_idx},
        )


class TextualGradientDescentMomentumOptimizer(BasePromptOptimizer):
    def estimate_metric_calls(self, batch_size: int) -> int:
        return batch_size * (2 + self.config.candidate_pool_size + self.config.bootstrap_samples)

    def _summarize_momentum_prompts(self, prompts: list[str]) -> str:
        if not prompts:
            return ""
        if len(prompts) == 1:
            return prompts[0]
        response = self.engine(
            "\n".join(
                [
                    "You are summarizing historically strong math-solving prompts.",
                    "Extract the durable prompt strategies that should be preserved.",
                    "Historical prompts:",
                    "\n\n".join(prompts),
                    f"Return only the summary between <{MOMENTUM_TAG}> and </{MOMENTUM_TAG}>.",
                ]
            )
        )
        return extract_tagged_text(response, MOMENTUM_TAG)

    def _sample_momentum_prompts(
        self,
        experiment: "AIMEExperiment",
        batch: list[Any],
        prompt_history: list[PromptHistoryEntry],
    ) -> tuple[list[str], list[dict[str, Any]], int]:
        recent_history = prompt_history[-self.config.momentum_buffer_size :]
        if not recent_history:
            return [], [], 0

        bootstrap_scores: list[float] = []
        metadata: list[dict[str, Any]] = []
        metric_calls = 0
        for entry in recent_history:
            score, calls = self._bootstrap_score(experiment, entry.prompt, batch)
            bootstrap_scores.append(score)
            metric_calls += calls
            metadata.append({"prompt": entry.prompt, "bootstrap_score": score, "step_idx": entry.step_idx})

        probs = softmax(bootstrap_scores)
        selected_indices = weighted_sample_without_replacement(
            list(range(len(recent_history))),
            probs,
            self.config.momentum_size,
            rng=self.rng,
        )
        return [recent_history[idx].prompt for idx in selected_indices], metadata, metric_calls

    def step(
        self,
        *,
        experiment: "AIMEExperiment",
        batch: list[Any],
        current_batch_evaluation: "BatchEvaluation",
        step_idx: int,
        prompt_history: list[PromptHistoryEntry],
    ) -> OptimizationStepArtifacts:
        gradient_signal = self._gradient_signal()

        candidate_prompts = self._generate_candidate_pool(gradient_signal)
        candidate_scores: list[float] = []
        metric_calls = current_batch_evaluation.metric_calls
        for prompt in candidate_prompts:
            score, calls = self._score_prompt(experiment, prompt, batch)
            candidate_scores.append(score)
            metric_calls += calls

        selected_indices = gumbel_topk_indices(
            candidate_scores,
            self.config.top_k,
            tau=self.config.gumbel_temperature,
            rng=self.rng,
        )
        selected_candidates = [candidate_prompts[idx] for idx in selected_indices] or [candidate_prompts[0]]

        momentum_prompts, momentum_metadata, momentum_metric_calls = self._sample_momentum_prompts(
            experiment,
            batch,
            prompt_history,
        )
        metric_calls += momentum_metric_calls
        momentum_summary = self._summarize_momentum_prompts(momentum_prompts)

        candidate_block = json.dumps(selected_candidates, ensure_ascii=False, indent=2)
        extra_context = "\n".join(
            [
                "Candidate prompts selected by Gumbel-Top-k:",
                candidate_block,
                f"Momentum summary:\n{momentum_summary}" if momentum_summary else "",
                "Synthesize the strongest ideas from the selected candidates and momentum context.",
            ]
        )
        new_prompt = self._update_prompt(gradient_signal, extra_context=extra_context)
        updated_score, updated_metric_calls = self._score_prompt(experiment, new_prompt, batch)
        bootstrap_score, bootstrap_metric_calls = self._bootstrap_score(experiment, new_prompt, batch)
        metric_calls += updated_metric_calls + bootstrap_metric_calls
        self.variable.set_value(new_prompt)

        return OptimizationStepArtifacts(
            prompt=new_prompt,
            current_batch_score=current_batch_evaluation.average_score,
            updated_batch_score=updated_score,
            combined_gradient=gradient_signal,
            metric_calls=metric_calls,
            metadata={
                "algorithm": "tsgd_m",
                "step_idx": step_idx,
                "candidate_scores": [
                    {"prompt": prompt, "score": score} for prompt, score in zip(candidate_prompts, candidate_scores, strict=True)
                ],
                "selected_candidates": selected_candidates,
                "momentum_prompts": momentum_prompts,
                "momentum_metadata": momentum_metadata,
                "bootstrap_score": bootstrap_score,
            },
        )


def build_prompt_optimizer(
    *,
    variable: PromptVariable,
    engine,
    config: TextGradAlgorithmConfig,
    rng: random.Random,
) -> BasePromptOptimizer:
    if config.algorithm == "tgd":
        return TextualGradientDescentOptimizer(variable=variable, engine=engine, config=config, rng=rng)
    if config.algorithm == "tsgd_m":
        return TextualGradientDescentMomentumOptimizer(variable=variable, engine=engine, config=config, rng=rng)
    raise ValueError(
        f"Unsupported AIME_TEXTGRAD_ALGORITHM={config.algorithm!r}. Supported values are 'tgd' and 'tsgd_m'."
    )
