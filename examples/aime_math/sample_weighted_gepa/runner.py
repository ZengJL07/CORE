from __future__ import annotations

import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

from examples.aime_math.experiment import AIMEExperiment
from examples.aime_math.prompt_optimizers.sampling import softmax, weighted_sample_without_replacement
from examples.aime_math.sample_weighted_gepa.config import SampleWeightedGEPAConfig
from examples.aime_math.sample_weighted_gepa.reflection import (
    build_reflection_prompt,
    extract_reflection_and_prompt,
    format_batch_feedback,
)
from examples.aime_math.sample_weighted_gepa.state import (
    CandidateRecord,
    InnerLoopAttempt,
    OuterStepRecord,
    SampleWeightedGEPARunResult,
)


class SampleWeightedGEPARunner:
    def __init__(self, experiment: AIMEExperiment, config: SampleWeightedGEPAConfig):
        self.experiment = experiment
        self.config = config
        self.rng = random.Random(experiment.config.seed)
        self.trainset = list(experiment.trainset)
        self.valset = list(experiment.valset)
        self.total_metric_calls = 0
        self._next_candidate_id = 0

        self.sampled_count: dict[str, int] = {self._sample_id(example): 0 for example in self.trainset}
        self.candidate_pool: list[CandidateRecord] = []
        self.outer_step_records: list[OuterStepRecord] = []
        self.candidate_sampled_count: dict[int, int] = {}

    def run(self) -> SampleWeightedGEPARunResult:
        print(
            "[AIME] SampleWeightedGEPA config: "
            f"candidate_pool_size={self.config.candidate_pool_size}, "
            f"reflect_train_batch_size={self.config.reflect_train_batch_size}, "
            f"probe_val_batch_size={self.config.probe_val_batch_size}, "
            f"inner_steps={self.config.inner_steps}, "
            f"max_outer_steps={self.config.max_outer_steps}, "
            f"all_fail_bonus={self.config.all_fail_bonus}, "
            f"score_sampling_temperature={self.config.score_sampling_temperature}"
        )

        initial_candidate = self._build_candidate_record(
            prompt=self.experiment.config.initial_prompt,
            parent_id=None,
        )
        self.candidate_pool.append(initial_candidate)
        print(f"[AIME] Initial validation score: {initial_candidate.val_score:.2%}")

        for outer_step in range(1, self.config.max_outer_steps + 1):
            if self.total_metric_calls >= self.experiment.config.max_metric_calls:
                print(
                    f"[AIME] Stopping before outer step {outer_step}: "
                    f"metric budget exhausted ({self.total_metric_calls}/{self.experiment.config.max_metric_calls})."
                )
                break

            sample_weights = self._compute_train_sample_weights()
            reflect_indices = self._sample_train_indices(sample_weights)
            probe_indices = self._sample_val_indices()
            if not reflect_indices or not probe_indices:
                print(f"[AIME] Stopping at outer step {outer_step}: empty sampled batch.")
                break

            reflect_batch = [self.trainset[idx] for idx in reflect_indices]
            probe_batch = [self.valset[idx] for idx in probe_indices]
            mixed_batch = reflect_batch + probe_batch

            base_candidate = self._sample_base_candidate()
            base_eval = self._evaluate_batch(base_candidate.prompt, mixed_batch)
            base_mixed_score = base_eval.average_score

            current_prompt = base_candidate.prompt
            inner_attempts: list[InnerLoopAttempt] = [
                InnerLoopAttempt(
                    step_idx=0,
                    prompt=base_candidate.prompt,
                    reflection="",
                    train_score=float("nan"),
                )
            ]

            for inner_step in range(1, self.config.inner_steps + 1):
                train_eval = self._evaluate_batch(current_prompt, reflect_batch)
                train_score = train_eval.average_score
                previous_attempt = inner_attempts[-1]
                inner_attempts[-1] = InnerLoopAttempt(
                    step_idx=previous_attempt.step_idx,
                    prompt=previous_attempt.prompt,
                    reflection=previous_attempt.reflection,
                    train_score=train_score,
                    mixed_score=previous_attempt.mixed_score,
                )

                if all(item.score >= 1.0 for item in train_eval.example_evaluations):
                    print(
                        f"[AIME] Outer step {outer_step}, inner step {inner_step}: "
                        "reflect batch solved perfectly, stopping inner loop early."
                    )
                    break

                prompt = build_reflection_prompt(
                    current_prompt=current_prompt,
                    current_batch_feedback=format_batch_feedback(train_eval),
                    current_batch_score=train_score,
                    previous_attempts=inner_attempts,
                )
                response = self.experiment.reflection_lm(prompt)
                reflection, mutated_prompt = extract_reflection_and_prompt(response)
                if not mutated_prompt:
                    mutated_prompt = current_prompt

                current_prompt = mutated_prompt
                inner_attempts.append(
                    InnerLoopAttempt(
                        step_idx=inner_step,
                        prompt=mutated_prompt,
                        reflection=reflection,
                        train_score=float("nan"),
                    )
                )

            final_prompt = current_prompt
            new_batch_eval = self._evaluate_batch(final_prompt, mixed_batch)
            candidate_mixed_score = new_batch_eval.average_score
            accepted_to_pool = False
            candidate_val_score: float | None = None
            is_new_prompt = not self._pool_contains_prompt(final_prompt)

            if is_new_prompt and candidate_mixed_score > base_mixed_score:
                candidate_val_score = self._evaluate_val_score(final_prompt)
                accepted_to_pool = self._maybe_insert_candidate(
                    prompt=final_prompt,
                    val_score=candidate_val_score,
                    parent_id=base_candidate.candidate_id,
                )

            self._increment_sample_counts(reflect_batch)
            self.outer_step_records.append(
                OuterStepRecord(
                    outer_step=outer_step,
                    base_candidate_id=base_candidate.candidate_id,
                    base_prompt=base_candidate.prompt,
                    reflect_train_ids=[self._sample_id(example) for example in reflect_batch],
                    probe_val_ids=[self._sample_id(example) for example in probe_batch],
                    base_mixed_score=base_mixed_score,
                    candidate_mixed_score=candidate_mixed_score,
                    candidate_val_score=candidate_val_score,
                    accepted_to_pool=accepted_to_pool,
                    total_metric_calls=self.total_metric_calls,
                    inner_attempts=inner_attempts,
                    metadata={"pool_candidate_ids": [candidate.candidate_id for candidate in self.candidate_pool]},
                )
            )

            print(
                f"[AIME] Outer step {outer_step}: "
                f"base_score={base_mixed_score:.2%}, "
                f"candidate_score={candidate_mixed_score:.2%}, "
                f"accepted={accepted_to_pool}, "
                f"pool_best={self._best_candidate().val_score:.2%}, "
                f"metric_calls={self.total_metric_calls}"
            )

            if self.total_metric_calls >= self.experiment.config.max_metric_calls:
                print(
                    f"[AIME] Metric budget reached after outer step {outer_step}: "
                    f"{self.total_metric_calls}/{self.experiment.config.max_metric_calls}"
                )
                break

        best_candidate = self._best_candidate()
        self._write_summary(best_candidate)
        return SampleWeightedGEPARunResult(
            best_prompt=best_candidate.prompt,
            best_val_score=best_candidate.val_score,
            total_metric_calls=self.total_metric_calls,
            total_outer_steps=len(self.outer_step_records),
            candidate_pool=list(self.candidate_pool),
            outer_step_records=list(self.outer_step_records),
        )

    def _sample_id(self, example: Any) -> str:
        return str(getattr(example, "input", None) or example.input)

    def _evaluate_batch(self, prompt: str, batch: list[Any]):
        evaluation = self.experiment.evaluate_prompt_on_batch(prompt, batch)
        self.total_metric_calls += evaluation.metric_calls
        return evaluation

    def _evaluate_val_score(self, prompt: str) -> float:
        stats = self.experiment.evaluate_prompt_summary(
            prompt,
            self.valset,
            pass_k=1,
            cache_label=f"{self.experiment.config.backend}_val_candidate_{self._next_candidate_id}",
        )
        self.total_metric_calls += int(stats["total_attempts"])
        return float(stats["mean_score"])

    def _evaluate_full_train_scores(self, prompt: str) -> dict[str, float]:
        batch_eval = self._evaluate_batch(prompt, self.trainset)
        return {self._sample_id(item.example): item.score for item in batch_eval.example_evaluations}

    def _build_candidate_record(self, *, prompt: str, parent_id: int | None) -> CandidateRecord:
        val_score = self._evaluate_val_score(prompt)
        train_scores_by_sample = self._evaluate_full_train_scores(prompt)
        candidate = CandidateRecord(
            candidate_id=self._next_candidate_id,
            prompt=prompt,
            val_score=val_score,
            parent_id=parent_id,
            train_scores_by_sample=train_scores_by_sample,
        )
        self._next_candidate_id += 1
        return candidate

    def _compute_train_sample_weights(self) -> list[float]:
        weights: list[float] = []
        for example in self.trainset:
            sample_id = self._sample_id(example)
            sample_scores = [
                candidate.train_scores_by_sample.get(sample_id, 0.0)
                for candidate in self.candidate_pool
            ]
            mean_score = sum(sample_scores) / len(sample_scores)
            variance = sum((score - mean_score) ** 2 for score in sample_scores) / len(sample_scores)
            all_fail = all(score <= 0.0 for score in sample_scores)
            weight = variance + (self.config.all_fail_bonus if all_fail else 0.0)
            weight /= math.sqrt(1 + self.sampled_count[sample_id])
            weights.append(max(weight, 0.0))
        return weights

    def _sample_train_indices(self, weights: list[float]) -> list[int]:
        population = list(range(len(self.trainset)))
        return weighted_sample_without_replacement(
            population,
            weights,
            self.config.reflect_train_batch_size,
            rng=self.rng,
        )

    def _sample_val_indices(self) -> list[int]:
        count = min(self.config.probe_val_batch_size, len(self.valset))
        return self.rng.sample(list(range(len(self.valset))), count)

    def _sample_base_candidate(self) -> CandidateRecord:
        scores = [candidate.val_score for candidate in self.candidate_pool]
        probs = softmax(scores, temperature=self.config.score_sampling_temperature)

        penalized_probs = []
        for candidate, prob in zip(self.candidate_pool, probs, strict=True):
            count = self.candidate_sampled_count.get(candidate.candidate_id, 0)
            penalized_probs.append(prob / math.sqrt(1 + count))

        total_prob = sum(penalized_probs)
        if total_prob > 0:
            probs = [p / total_prob for p in penalized_probs]
        else:
            probs = [1.0 / len(self.candidate_pool)] * len(self.candidate_pool)

        draw = self.rng.random()
        running = 0.0
        selected_candidate = self.candidate_pool[-1]
        for candidate, prob in zip(self.candidate_pool, probs, strict=True):
            running += prob
            if draw <= running:
                selected_candidate = candidate
                break

        self.candidate_sampled_count[selected_candidate.candidate_id] = self.candidate_sampled_count.get(selected_candidate.candidate_id, 0) + 1
        return selected_candidate

    def _maybe_insert_candidate(self, *, prompt: str, val_score: float, parent_id: int | None) -> bool:
        for candidate in self.candidate_pool:
            if candidate.prompt == prompt:
                return False

        if len(self.candidate_pool) >= self.config.candidate_pool_size:
            worst_score = min(candidate.val_score for candidate in self.candidate_pool)
            if val_score <= worst_score:
                return False

        train_scores_by_sample = self._evaluate_full_train_scores(prompt)
        candidate = CandidateRecord(
            candidate_id=self._next_candidate_id,
            prompt=prompt,
            val_score=val_score,
            parent_id=parent_id,
            train_scores_by_sample=train_scores_by_sample,
        )
        self._next_candidate_id += 1
        self.candidate_pool.append(candidate)
        self.candidate_pool.sort(key=lambda item: item.val_score, reverse=True)
        self.candidate_pool = self.candidate_pool[: self.config.candidate_pool_size]
        return any(item.candidate_id == candidate.candidate_id for item in self.candidate_pool)

    def _increment_sample_counts(self, batch: list[Any]) -> None:
        for example in batch:
            sample_id = self._sample_id(example)
            self.sampled_count[sample_id] += 1

    def _best_candidate(self) -> CandidateRecord:
        return max(self.candidate_pool, key=lambda candidate: candidate.val_score)

    def _pool_contains_prompt(self, prompt: str) -> bool:
        return any(candidate.prompt == prompt for candidate in self.candidate_pool)

    def _write_summary(self, best_candidate: CandidateRecord) -> None:
        summary = {
            "algorithm": "sample_weighted_gepa",
            "best_prompt": best_candidate.prompt,
            "best_val_score": best_candidate.val_score,
            "total_metric_calls": self.total_metric_calls,
            "total_outer_steps": len(self.outer_step_records),
            "candidate_pool": [asdict(candidate) for candidate in self.candidate_pool],
            "outer_step_records": [asdict(record) for record in self.outer_step_records],
        }
        Path(self.experiment.config.run_dir).mkdir(parents=True, exist_ok=True)
        summary_path = Path(self.experiment.config.run_dir) / "sample_weighted_gepa_summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
