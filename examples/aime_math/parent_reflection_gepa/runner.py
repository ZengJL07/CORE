from __future__ import annotations

import json
import math
import random
import threading
from statistics import mean, pstdev
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from examples.aime_math.parent_reflection_gepa.config import ParentReflectionGEPAConfig
from examples.aime_math.parent_reflection_gepa.reflection import (
    build_reflection_prompt,
    extract_reflection_and_prompt,
    format_batch_feedback,
)
from examples.aime_math.parent_reflection_gepa.state import (
    BranchRecord,
    CandidatePoint,
    CandidateRecord,
    OuterStepRecord,
    ParentReflectionGEPARunResult,
)
from examples.aime_math.prompt_optimizers.sampling import softmax

if TYPE_CHECKING:
    from examples.aime_math.experiment import AIMEExperiment


class ParentReflectionGEPARunner:
    def __init__(self, experiment: AIMEExperiment, config: ParentReflectionGEPAConfig):
        self.experiment = experiment
        self.config = config
        self.rng = random.Random(experiment.config.seed)
        self.trainset = list(experiment.trainset)
        self.valset = list(experiment.valset)
        self.total_metric_calls = 0
        self._next_candidate_id = 0
        self._metric_lock = threading.Lock()

        self.candidate_pool: list[CandidateRecord] = []
        self.outer_step_records: list[OuterStepRecord] = []
        self.candidate_points: list[CandidatePoint] = []
        self._val_ids = [self._sample_id(example) for example in self.valset]

    def _format_prompt_for_log(self, prompt: str) -> str:
        return repr(prompt)

    def _short_sample_id(self, sample_id: str, width: int = 72) -> str:
        compact = " ".join(sample_id.split())
        if len(compact) <= width:
            return compact
        return f"{compact[: width - 3]}..."

    def _log_pool_snapshot(self) -> None:
        print("[AIME] Candidate pool snapshot:")
        for candidate in self.candidate_pool:
            parent_display = "root" if candidate.parent_id is None else str(candidate.parent_id)
            print(
                "[AIME]   "
                f"id={candidate.candidate_id} "
                f"parent={parent_display} "
                f"val={candidate.val_score:.2%} "
                f"selected={candidate.times_selected_as_current} "
                f"prompt={self._format_prompt_for_log(candidate.prompt)}"
            )

    def run(self) -> ParentReflectionGEPARunResult:
        print(
            "[AIME] ParentReflectionGEPA config: "
            f"candidate_pool_size={self.config.candidate_pool_size}, "
            f"reflect_train_batch_size={self.config.reflect_train_batch_size}, "
            f"probe_val_batch_size={self.config.probe_val_batch_size}, "
            f"parallel_branches={self.config.num_parallel_branches}, "
            f"max_outer_steps={self.config.max_outer_steps}, "
            f"score_sampling_temperature={self.config.score_sampling_temperature}, "
            f"ucb_exploration_coef={self.config.ucb_exploration_coef}, "
            f"prompt_ucb_exploration_coef={self.config.prompt_ucb_exploration_coef}, "
            f"prompt_score_std_floor={self.config.prompt_score_std_floor}, "
            f"train_retries={self.config.train_rejection_max_retries}, "
            f"val_retries={self.config.val_rejection_max_retries}, "
            f"use_parent_history={self.config.use_parent_history}"
        )

        initial_candidate = self._build_candidate_record(
            prompt=self.experiment.config.initial_prompt,
            parent_candidate=None,
            reflection_from_parent="",
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

            print(f"[AIME] ===== Outer step {outer_step} =====")
            self._log_pool_snapshot()

            branch_records = self._run_parallel_branches(outer_step=outer_step)
            inserted_candidate_ids = self._evaluate_and_insert_survivors(
                branch_records=branch_records,
                outer_step=outer_step,
            )

            self.outer_step_records.append(
                OuterStepRecord(
                    outer_step=outer_step,
                    sampled_current_candidate_ids=[branch.current_candidate_id for branch in branch_records],
                    inserted_candidate_ids=inserted_candidate_ids,
                    branch_records=branch_records,
                    total_metric_calls=self.total_metric_calls,
                    metadata={
                        "pool_candidate_ids": [candidate.candidate_id for candidate in self.candidate_pool],
                        "num_survivors": sum(1 for branch in branch_records if branch.passed_total_gate),
                    },
                )
            )

            print(
                f"[AIME] Outer step {outer_step}: "
                f"current_ids={[branch.current_candidate_id for branch in branch_records]}, "
                f"survivors={sum(1 for branch in branch_records if branch.passed_total_gate)}, "
                f"inserted={len(inserted_candidate_ids)}, "
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
        return ParentReflectionGEPARunResult(
            best_prompt=best_candidate.prompt,
            best_val_score=best_candidate.val_score,
            total_metric_calls=self.total_metric_calls,
            total_outer_steps=len(self.outer_step_records),
            candidate_pool=list(self.candidate_pool),
            outer_step_records=list(self.outer_step_records),
            candidate_points=list(self.candidate_points),
        )

    def _sample_id(self, example: Any) -> str:
        return str(getattr(example, "input", None) or example.input)

    def _evaluate_batch(self, prompt: str, batch: list[Any]):
        evaluation = self.experiment.evaluate_prompt_on_batch(prompt, batch)
        with self._metric_lock:
            self.total_metric_calls += evaluation.metric_calls
        return evaluation

    def _evaluate_full_val(self, prompt: str) -> tuple[float, dict[str, float]]:
        print(f"[AIME] Full validation start: prompt={self._format_prompt_for_log(prompt)}")
        batch_eval = self._evaluate_batch(prompt, self.valset)
        val_scores = {
            self._sample_id(item.example): item.score
            for item in batch_eval.example_evaluations
        }
        print(
            "[AIME] Full validation done: "
            f"score={batch_eval.average_score:.2%} "
            f"examples={len(batch_eval.example_evaluations)}"
        )
        return batch_eval.average_score, val_scores

    def _build_candidate_record(
        self,
        *,
        prompt: str,
        parent_candidate: CandidateRecord | None,
        reflection_from_parent: str,
    ) -> CandidateRecord:
        val_score, val_scores_by_sample = self._evaluate_full_val(prompt)
        candidate = CandidateRecord(
            candidate_id=self._next_candidate_id,
            prompt=prompt,
            val_score=val_score,
            parent_id=None if parent_candidate is None else parent_candidate.candidate_id,
            parent_prompt=None if parent_candidate is None else parent_candidate.prompt,
            reflection_from_parent=reflection_from_parent,
            val_scores_by_sample=val_scores_by_sample,
            train_scores_by_sample={} if parent_candidate is None else dict(parent_candidate.train_scores_by_sample),
        )
        self._next_candidate_id += 1
        print(
            "[AIME] Built candidate record: "
            f"id={candidate.candidate_id} "
            f"parent={candidate.parent_id} "
            f"val={val_score:.2%} "
            f"prompt={self._format_prompt_for_log(prompt)}"
        )
        return candidate

    def _compute_prompt_ucb_components(
        self,
        candidates: list[CandidateRecord],
    ) -> list[tuple[CandidateRecord, float, float, float]]:
        if not candidates:
            return []

        scores = [candidate.val_score for candidate in candidates]
        score_mean = mean(scores)
        score_std = pstdev(scores)
        effective_std = max(score_std, self.config.prompt_score_std_floor)
        total_selected = sum(candidate.times_selected_as_current for candidate in self.candidate_pool)
        log_term = math.log(total_selected + 1.0)

        components: list[tuple[CandidateRecord, float, float, float]] = []
        for candidate in candidates:
            advantage = (candidate.val_score - score_mean) / effective_std
            exploration = self.config.prompt_ucb_exploration_coef * math.sqrt(
                log_term / (candidate.times_selected_as_current + 1.0)
            )
            ucb_score = advantage + exploration
            components.append((candidate, advantage, exploration, ucb_score))
        return components

    def _sample_current_candidates_with_prompt_ucb(self, outer_step: int) -> list[CandidateRecord]:
        pool_limited_cap = len(self.candidate_pool) + 1
        target_count = min(self.config.num_parallel_branches, outer_step, pool_limited_cap)
        if not self.candidate_pool or target_count <= 0:
            return []

        selected_candidates: list[CandidateRecord] = []
        print(
            "[AIME] Sampling current prompts for parallel branches with prompt-UCB softmax: "
            f"target={self.config.num_parallel_branches} "
            f"pool_cap={pool_limited_cap} "
            f"actual={target_count}"
        )
        for branch_slot in range(1, target_count + 1):
            components = self._compute_prompt_ucb_components(self.candidate_pool)
            ucb_scores = [ucb_score for _, _, _, ucb_score in components]
            probs = softmax(ucb_scores, temperature=self.config.score_sampling_temperature)
            draw = self.rng.random()
            running = 0.0
            selected = components[-1][0]
            for (candidate, _, _, _), prob in zip(components, probs, strict=True):
                running += prob
                if draw <= running:
                    selected = candidate
                    break

            selected.times_selected_as_current += 1
            selected_candidates.append(selected)

        return selected_candidates

    def _sample_train_batch_with_rejection(self, current_candidate: CandidateRecord, *, branch_idx: int | None = None):
        count = min(self.config.reflect_train_batch_size, len(self.trainset))
        last_batch: list[Any] = []
        last_ids: list[str] = []
        last_eval = None
        accepted_attempt = self.config.train_rejection_max_retries
        failing_indices = [
            idx
            for idx, example in enumerate(self.trainset)
            if current_candidate.train_scores_by_sample.get(self._sample_id(example), 0.0) < 1.0
        ]

        for attempt in range(1, self.config.train_rejection_max_retries + 1):
            if failing_indices:
                if len(failing_indices) >= count:
                    indices = self.rng.sample(failing_indices, count)
                else:
                    remaining_indices = [idx for idx in range(len(self.trainset)) if idx not in failing_indices]
                    extra = self.rng.sample(remaining_indices, count - len(failing_indices)) if remaining_indices else []
                    indices = list(failing_indices) + extra
            else:
                indices = self.rng.sample(list(range(len(self.trainset))), count)
            batch = [self.trainset[idx] for idx in indices]
            batch_eval = self._evaluate_batch(current_candidate.prompt, batch)
            last_batch = batch
            last_ids = [self._sample_id(example) for example in batch]
            last_eval = batch_eval
            for item in batch_eval.example_evaluations:
                current_candidate.train_scores_by_sample[self._sample_id(item.example)] = item.score
            if batch_eval.average_score < 1.0:
                accepted_attempt = attempt
                break

        assert last_eval is not None
        print(
            "[AIME] Train batch sampling"
            f"{f' [branch {branch_idx}]' if branch_idx is not None else ''}: "
            f"attempt={accepted_attempt}/{self.config.train_rejection_max_retries} "
            f"score={last_eval.average_score:.2%} "
            f"failing_pool={len(failing_indices)} "
            f"ids={[self._short_sample_id(sample_id, 48) for sample_id in last_ids]}"
        )
        return last_batch, last_ids, last_eval

    def _compute_pool_variance_norms(self) -> dict[str, float]:
        variances: dict[str, float] = {}
        for sample_id in self._val_ids:
            scores = [candidate.val_scores_by_sample.get(sample_id, 0.0) for candidate in self.candidate_pool]
            if len(scores) <= 1:
                variances[sample_id] = 0.0
                continue
            mean_score = sum(scores) / len(scores)
            variances[sample_id] = sum((score - mean_score) ** 2 for score in scores) / len(scores)

        max_variance = max(variances.values(), default=0.0)
        if max_variance <= 0.0:
            return {sample_id: 0.0 for sample_id in self._val_ids}
        return {sample_id: value / max_variance for sample_id, value in variances.items()}

    def _compute_ucb_components(self, current_candidate: CandidateRecord) -> list[tuple[str, float, float, float, float]]:
        variance_norms = self._compute_pool_variance_norms()
        log_term = math.log(current_candidate.times_selected_as_current + 1.0)
        components: list[tuple[str, float, float, float, float]] = []
        for example in self.valset:
            sample_id = self._sample_id(example)
            err = 1.0 if current_candidate.val_scores_by_sample.get(sample_id, 0.0) < 1.0 else 0.0
            count = current_candidate.val_sampled_count.get(sample_id, 0)
            exploration = self.config.ucb_exploration_coef * math.sqrt(log_term / (count + 1.0))
            variance = variance_norms.get(sample_id, 0.0)
            weight = variance + err + exploration
            components.append((sample_id, variance, err, exploration, weight))
        return components

    def _weighted_sample_indices_with_replacement(self, weights: list[float], count: int) -> list[int]:
        if count <= 0:
            return []
        positive_weights = [max(weight, 0.0) for weight in weights]
        if sum(positive_weights) <= 0:
            return [self.rng.randrange(len(weights)) for _ in range(count)]
        return self.rng.choices(list(range(len(weights))), weights=positive_weights, k=count)

    def _probe_batch_score_from_pool(
        self,
        current_candidate: CandidateRecord,
        probe_batch: list[Any],
    ) -> float:
        if not probe_batch:
            return 0.0
        total = sum(
            current_candidate.val_scores_by_sample.get(self._sample_id(example), 0.0)
            for example in probe_batch
        )
        return total / len(probe_batch)

    def _sample_probe_val_batch_with_rejection(
        self,
        current_candidate: CandidateRecord,
        *,
        branch_idx: int | None = None,
    ):
        count = min(self.config.probe_val_batch_size, len(self.valset))
        if count == 0:
            print(
                "[AIME] Probe val batch sampling"
                f"{f' [branch {branch_idx}]' if branch_idx is not None else ''}: "
                "disabled (probe_val_batch_size=0)"
            )
            return [], []

        ucb_components = self._compute_ucb_components(current_candidate)
        weights = [weight for _, _, _, _, weight in ucb_components]
        last_batch: list[Any] = []
        last_ids: list[str] = []
        accepted_attempt = self.config.val_rejection_max_retries
        final_batch_score = 0.0

        for attempt in range(1, self.config.val_rejection_max_retries + 1):
            indices = self._weighted_sample_indices_with_replacement(weights, count)
            batch = [self.valset[idx] for idx in indices]
            batch_score = self._probe_batch_score_from_pool(current_candidate, batch)
            last_batch = batch
            last_ids = [self._sample_id(example) for example in batch]
            final_batch_score = batch_score
            if batch_score < 1.0:
                accepted_attempt = attempt
                break

        for sample_id in last_ids:
            current_candidate.val_sampled_count[sample_id] = current_candidate.val_sampled_count.get(sample_id, 0) + 1

        print(
            "[AIME] Probe val batch sampling"
            f"{f' [branch {branch_idx}]' if branch_idx is not None else ''}: "
            f"attempt={accepted_attempt}/{self.config.val_rejection_max_retries} "
            f"score={final_batch_score:.2%} "
            f"ids={[self._short_sample_id(sample_id, 48) for sample_id in last_ids]}"
        )
        return last_batch, last_ids

    def _compute_base_total_score(self, *, train_eval, current_candidate: CandidateRecord, probe_batch: list[Any]) -> float:
        train_total = sum(item.score for item in train_eval.example_evaluations)
        probe_total = sum(
            current_candidate.val_scores_by_sample.get(self._sample_id(example), 0.0)
            for example in probe_batch
        )
        total_count = len(train_eval.example_evaluations) + len(probe_batch)
        if total_count == 0:
            return 0.0
        return (train_total + probe_total) / total_count

    def _run_parallel_branches(
        self,
        *,
        outer_step: int,
    ) -> list[BranchRecord]:
        sampled_current_candidates = self._sample_current_candidates_with_prompt_ucb(outer_step)
        if not sampled_current_candidates:
            return []

        max_workers = min(len(sampled_current_candidates), max(1, self.experiment.config.max_workers))
        branch_contexts: list[dict[str, Any]] = []
        print(
            "[AIME] Preparing branch contexts: "
            f"num_branches={len(sampled_current_candidates)} "
            f"max_workers={max_workers}"
        )
        for branch_idx, current_candidate in enumerate(sampled_current_candidates, start=1):
            train_batch, train_ids, base_train_eval = self._sample_train_batch_with_rejection(
                current_candidate,
                branch_idx=branch_idx,
            )
            probe_batch, probe_ids = self._sample_probe_val_batch_with_rejection(
                current_candidate,
                branch_idx=branch_idx,
            )
            total_batch = train_batch + probe_batch
            base_train_score = base_train_eval.average_score
            base_total_score = self._compute_base_total_score(
                train_eval=base_train_eval,
                current_candidate=current_candidate,
                probe_batch=probe_batch,
            )
            print(
                f"[AIME] Branch {branch_idx} base scores: "
                f"train={base_train_score:.2%} total={base_total_score:.2%}"
            )
            branch_contexts.append(
                {
                    "branch_idx": branch_idx,
                    "current_candidate": current_candidate,
                    "train_batch": train_batch,
                    "train_batch_ids": train_ids,
                    "probe_batch_ids": probe_ids,
                    "base_train_eval": base_train_eval,
                    "total_batch": total_batch,
                    "base_train_score": base_train_score,
                    "base_total_score": base_total_score,
                }
            )

        branch_records: list[BranchRecord] = []
        print("[AIME] Launching parallel branch optimization.")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self._run_branch,
                    outer_step=outer_step,
                    branch_idx=context["branch_idx"],
                    current_candidate=context["current_candidate"],
                    train_batch=context["train_batch"],
                    train_batch_ids=context["train_batch_ids"],
                    probe_batch_ids=context["probe_batch_ids"],
                    base_train_eval=context["base_train_eval"],
                    total_batch=context["total_batch"],
                    base_train_score=context["base_train_score"],
                    base_total_score=context["base_total_score"],
                ): context["branch_idx"]
                for context in branch_contexts
            }
            for future in as_completed(futures):
                branch_records.append(future.result())

        branch_records.sort(key=lambda record: record.branch_idx)
        print("[AIME] Branch outcomes:")
        for record in branch_records:
            total_display = f"{record.total_score:.2%}" if record.total_score is not None else "None"
            print(
                "[AIME]   "
                f"branch={record.branch_idx} "
                f"train={record.train_score:.2%} "
                f"train_gate={record.passed_train_gate} "
                f"total={total_display}"
            )
            print(
                "[AIME]     "
                f"total_gate={record.passed_total_gate} "
                f"prompt={self._format_prompt_for_log(record.final_prompt)}"
            )
        return branch_records

    def _run_branch(
        self,
        *,
        outer_step: int,
        branch_idx: int,
        current_candidate: CandidateRecord,
        train_batch: list[Any],
        train_batch_ids: list[str],
        probe_batch_ids: list[str],
        base_train_eval,
        total_batch: list[Any],
        base_train_score: float,
        base_total_score: float,
    ) -> BranchRecord:
        base_prompt = current_candidate.prompt
        print(
            f"[AIME] Branch {branch_idx} start: "
            f"current_id={current_candidate.candidate_id} "
            f"base_train={base_train_score:.2%} "
            f"base_total={base_total_score:.2%} "
            f"prompt={self._format_prompt_for_log(base_prompt)}"
        )

        if all(item.score >= 1.0 for item in base_train_eval.example_evaluations):
            print(f"[AIME] Branch {branch_idx}: train batch already solved perfectly, skipping mutation.")
            return BranchRecord(
                branch_idx=branch_idx,
                current_candidate_id=current_candidate.candidate_id,
                current_prompt=current_candidate.prompt,
                parent_prompt_used=current_candidate.parent_prompt,
                parent_reflection_used=current_candidate.reflection_from_parent,
                train_batch_ids=train_batch_ids,
                probe_val_ids=probe_batch_ids,
                base_train_score=base_train_score,
                base_total_score=base_total_score,
                generated_reflection="",
                final_prompt=base_prompt,
                train_score=base_train_score,
                total_score=base_train_score if not probe_batch_ids else None,
                passed_train_gate=False,
                passed_total_gate=False,
                metadata={
                    "skipped_reflection_on_train_perfect": True,
                    "probe_val_batch_size": len(probe_batch_ids),
                    "used_parent_context": False,
                    "mutated_prompt_changed": False,
                },
            )

        print(f"[AIME] Branch {branch_idx}: requesting parent-reflection mutation.")
        include_parent_history = self.config.use_parent_history
        prompt = build_reflection_prompt(
            current_prompt=base_prompt,
            current_batch_feedback=format_batch_feedback(base_train_eval),
            current_batch_score=base_train_score,
            parent_prompt=current_candidate.parent_prompt,
            parent_reflection=current_candidate.reflection_from_parent,
            include_parent_history=include_parent_history,
            branch_hint=f"outer-step {outer_step}, branch {branch_idx}",
        )
        response = self.experiment.reflection_lm(prompt)
        generated_reflection, mutated_prompt = extract_reflection_and_prompt(response)
        if not mutated_prompt:
            mutated_prompt = base_prompt
        print(
            f"[AIME] Branch {branch_idx}: "
            f"mutation_received changed={mutated_prompt != base_prompt} "
            f"next_prompt={self._format_prompt_for_log(mutated_prompt)}"
        )

        train_eval = self._evaluate_batch(mutated_prompt, train_batch)
        train_score = train_eval.average_score
        passed_train_gate = train_score > base_train_score
        print(
            f"[AIME] Branch {branch_idx}: "
            f"train_gate check current={train_score:.2%} "
            f"base={base_train_score:.2%} "
            f"passed={passed_train_gate}"
        )

        total_score: float | None = None
        passed_total_gate = False
        reused_train_gate_for_total = False

        if passed_train_gate:
            if not probe_batch_ids:
                total_score = train_score
                passed_total_gate = True
                reused_train_gate_for_total = True
                print(
                    f"[AIME] Branch {branch_idx}: "
                    "probe val batch is empty, reusing train gate result for total gate."
                )
            else:
                print(f"[AIME] Branch {branch_idx}: evaluating mixed batch gate.")
                total_eval = self._evaluate_batch(mutated_prompt, total_batch)
                total_score = total_eval.average_score
                passed_total_gate = total_score > base_total_score
                print(
                    f"[AIME] Branch {branch_idx}: "
                    f"total_gate check current={total_score:.2%} "
                    f"base={base_total_score:.2%} "
                    f"passed={passed_total_gate}"
                )
        else:
            print(f"[AIME] Branch {branch_idx}: skipped mixed batch gate because train gate failed.")

        return BranchRecord(
            branch_idx=branch_idx,
            current_candidate_id=current_candidate.candidate_id,
            current_prompt=current_candidate.prompt,
            parent_prompt_used=current_candidate.parent_prompt,
            parent_reflection_used=current_candidate.reflection_from_parent,
            train_batch_ids=train_batch_ids,
            probe_val_ids=probe_batch_ids,
            base_train_score=base_train_score,
            base_total_score=base_total_score,
            generated_reflection=generated_reflection,
            final_prompt=mutated_prompt,
            train_score=train_score,
            total_score=total_score,
            passed_train_gate=passed_train_gate,
            passed_total_gate=passed_total_gate,
            metadata={
                "probe_val_batch_size": len(probe_batch_ids),
                "used_parent_context": include_parent_history
                and bool(
                    (current_candidate.parent_prompt and current_candidate.parent_prompt.strip())
                    or current_candidate.reflection_from_parent.strip()
                ),
                "mutated_prompt_changed": mutated_prompt != base_prompt,
                "reused_train_gate_for_total": reused_train_gate_for_total,
            },
        )

    def _evaluate_and_insert_survivors(
        self,
        *,
        branch_records: list[BranchRecord],
        outer_step: int,
    ) -> list[int]:
        surviving_by_prompt: dict[str, BranchRecord] = {}
        for branch in branch_records:
            if not branch.passed_total_gate:
                continue
            existing = surviving_by_prompt.get(branch.final_prompt)
            if existing is None or (branch.total_score or float("-inf")) > (existing.total_score or float("-inf")):
                surviving_by_prompt[branch.final_prompt] = branch

        if not surviving_by_prompt:
            print("[AIME] No surviving branches passed both gates; candidate pool unchanged.")
            return []

        inserted_candidates: list[CandidateRecord] = []
        current_by_id = {candidate.candidate_id: candidate for candidate in self.candidate_pool}
        existing_by_prompt = {candidate.prompt: candidate for candidate in self.candidate_pool}
        print(f"[AIME] Evaluating survivors for pool update: count={len(surviving_by_prompt)}")

        for prompt, branch in surviving_by_prompt.items():
            if prompt in existing_by_prompt:
                existing = existing_by_prompt[prompt]
                branch.full_val_score = existing.val_score
                branch.accepted_to_pool = True
                print(
                    "[AIME] Survivor matches existing prompt: "
                    f"branch={branch.branch_idx} existing_id={existing.candidate_id} "
                    f"val={existing.val_score:.2%}"
                )
                continue

            parent_candidate = current_by_id.get(branch.current_candidate_id)
            if parent_candidate is None:
                raise ValueError(f"Missing current candidate {branch.current_candidate_id} while inserting survivor.")

            candidate = self._build_candidate_record(
                prompt=prompt,
                parent_candidate=parent_candidate,
                reflection_from_parent=branch.generated_reflection,
            )
            inserted_candidates.append(candidate)
            branch.full_val_score = candidate.val_score
            print(
                "[AIME] Full val for surviving branch: "
                f"branch={branch.branch_idx} new_id={candidate.candidate_id} "
                f"val={candidate.val_score:.2%} "
                f"prompt={self._format_prompt_for_log(candidate.prompt)}"
            )

        if not inserted_candidates:
            return []

        self.candidate_pool.extend(inserted_candidates)
        self.candidate_pool = sorted(
            self.candidate_pool,
            key=lambda candidate: (-candidate.val_score, candidate.candidate_id),
        )[: self.config.candidate_pool_size]
        print(
            "[AIME] Candidate pool trimmed to top scores: "
            f"size={len(self.candidate_pool)} "
            f"kept_ids={[candidate.candidate_id for candidate in self.candidate_pool]}"
        )

        kept_ids = {candidate.candidate_id for candidate in self.candidate_pool}
        inserted_ids: list[int] = []
        for candidate in inserted_candidates:
            kept = candidate.candidate_id in kept_ids
            inserted_ids.append(candidate.candidate_id)
            branch = surviving_by_prompt[candidate.prompt]
            branch.accepted_to_pool = kept
            self.candidate_points.append(
                CandidatePoint(
                    iteration=len(self.candidate_points) + 1,
                    outer_step=outer_step,
                    branch_idx=branch.branch_idx,
                    candidate_id=candidate.candidate_id,
                    score=candidate.val_score,
                    accepted_to_pool=kept,
                )
            )
            print(
                "[AIME] Pool update: "
                f"candidate_id={candidate.candidate_id} kept={kept} "
                f"val={candidate.val_score:.2%}"
            )

        return inserted_ids

    def _best_candidate(self) -> CandidateRecord:
        return max(self.candidate_pool, key=lambda candidate: candidate.val_score)

    def _write_summary(self, best_candidate: CandidateRecord) -> None:
        summary = {
            "algorithm": "parent_reflection_gepa",
            "best_prompt": best_candidate.prompt,
            "best_val_score": best_candidate.val_score,
            "total_metric_calls": self.total_metric_calls,
            "total_outer_steps": len(self.outer_step_records),
            "trainset_size": len(self.trainset),
            "valset_size": len(self.valset),
            "candidate_points": [asdict(point) for point in self.candidate_points],
            "candidate_pool": [asdict(candidate) for candidate in self.candidate_pool],
            "outer_step_records": [asdict(record) for record in self.outer_step_records],
        }
        Path(self.experiment.config.run_dir).mkdir(parents=True, exist_ok=True)
        summary_path = Path(self.experiment.config.run_dir) / "parent_reflection_gepa_summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
