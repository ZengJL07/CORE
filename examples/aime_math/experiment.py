from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from examples.aime_math.config import AIMEExperimentConfig
from examples.aime_math.utils import (
    build_dataset_task,
    CachedLanguageModel,
    configure_default_solver_client,
    generate_evaluation_plot,
    HMMT_FEB_2025_DATASET,
    HMMT_FEB_2025_TEST_JSONL,
    HMMT_FEB_2026_DATASET,
    HMMT_FEB_2026_TEST_JSONL,
    load_extra_math_testset,
)
from gepa.core.state import GEPAState
from gepa.optimize_anything import EngineConfig, GEPAConfig, MergeConfig, ReflectionConfig, SideInfo


@dataclass(frozen=True)
class ExampleEvaluation:
    example: Any
    score: float
    side_info: SideInfo


@dataclass(frozen=True)
class BatchEvaluation:
    prompt: str
    example_evaluations: list[ExampleEvaluation]

    @property
    def average_score(self) -> float:
        if not self.example_evaluations:
            return 0.0
        return sum(item.score for item in self.example_evaluations) / len(self.example_evaluations)

    @property
    def metric_calls(self) -> int:
        return len(self.example_evaluations)


class BestPromptPrinter:
    def __init__(self) -> None:
        self.best_score: float | None = None

    def on_valset_evaluated(self, event) -> None:
        score = float(event["average_score"])
        if event["candidate_idx"] == 0:
            self.best_score = score
            return

        if event["is_best_program"] and (self.best_score is None or score > self.best_score):
            self.best_score = score
            prompt = event["candidate"].get("current_candidate", str(event["candidate"]))
            print(
                "\n[AIME] New best prompt found "
                f"(candidate={event['candidate_idx']}, val_score={score:.2%}):\n{prompt}\n"
            )


class AIMEExperiment:
    def __init__(self, config: AIMEExperimentConfig):
        self.config = config
        self.task = build_dataset_task(
            config.dataset_name,
            mbpp_source=config.mbpp_source,
            mbpp_hf_dataset=config.mbpp_hf_dataset,
            mbpp_hf_config=config.mbpp_hf_config,
            mbpp_data_dir=config.mbpp_data_dir,
        )
        self._setup_solver()
        self.reflection_lm = self._build_reflection_lm()
        print("[AIME] Loading datasets...")
        splits = self.task.load_splits(seed=config.seed)
        self.trainset = list(splits.trainset[: config.max_train_examples]) if config.max_train_examples else list(splits.trainset)
        self.valset = list(splits.valset[: config.max_val_examples]) if config.max_val_examples else list(splits.valset)
        self.testset = list(splits.testset[: config.max_test_examples]) if config.max_test_examples else list(splits.testset)
        print(
            f"[AIME] Dataset ready: train={len(self.trainset)}, val={len(self.valset)}, test={len(self.testset)}"
        )

    def _setup_solver(self) -> None:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("Please set DEEPSEEK_API_KEY for the solver model.")

        configure_default_solver_client(
            cache_dir=self.config.solver_cache_dir,
            cache_namespace=self.config.solver_cache_namespace(),
            model_name=self.config.solver_model,
            completion_kwargs=self.config.solver_completion_kwargs(api_key),
            enable_cache=self.config.solver_cache_enabled,
            output_mode="python_code" if self.config.dataset_name == "mbpp" else "integer",
            task=self.task,
            api_max_retries=self.config.solver_api_max_retries,
        )
        print(f"[AIME] Solver API cache enabled: {self.config.solver_cache_enabled}")
        print(f"[AIME] Solver API cache directory: {self.config.solver_cache_dir}")
        print(f"[AIME] Final evaluation solver cache enabled: {self.config.eval_solver_cache_enabled}")
        print(f"[AIME] Final evaluation metric: pass@{self.config.eval_pass_k}")
        print(f"[AIME] Solver API max retries (transient failures): {self.config.solver_api_max_retries}")
        print("[AIME] Using direct request-level cache only; legacy cache migration is disabled.")

        print(
            "[AIME] Solver LM configured "
            f"(model={self.config.solver_model}, api_base={self.config.solver_api_base}, "
            f"temperature={self.config.solver_temperature}, max_tokens={self.config.solver_max_tokens})"
        )

    def _build_reflection_lm(self) -> CachedLanguageModel:
        reflection_api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not reflection_api_key:
            raise ValueError("Please set DEEPSEEK_API_KEY for the reflection model.")

        reflection_lm = CachedLanguageModel(
            self.config.reflection_model,
            cache_dir=self.config.reflection_cache_dir,
            api_max_retries=self.config.reflection_api_max_retries,
            api_key=reflection_api_key,
            api_base=self.config.solver_api_base,
            temperature=self.config.reflection_temperature,
        )
        print(
            "[AIME] Reflection LM configured "
            f"(model={self.config.reflection_model}, api_base={self.config.solver_api_base}, "
            f"temperature={self.config.reflection_temperature}, "
            f"api_max_retries={self.config.reflection_api_max_retries})"
        )
        print(f"[AIME] Reflection LM cache directory: {reflection_lm.cache_dir}")
        return reflection_lm

    def print_startup_banner(self) -> None:
        print("[AIME] Starting run...")
        print(f"[AIME] Backend: {self.config.backend}")
        print(f"[AIME] Seed: {self.config.seed} (override with AIME_SEED)")
        print(f"[AIME] Run artifacts will be written under: {self.config.run_dir}")
        print(f"[AIME] Best candidate strategy: {self.config.best_candidate_strategy}")
        print(f"[AIME] Skip baseline evaluation: {self.config.skip_baseline_eval}")
        if self.config.evaluate_existing_run_dir is not None:
            print(f"[AIME] Existing run evaluation source: {self.config.evaluate_existing_run_dir}")
        if self.config.evaluate_candidate_idx is not None:
            print(f"[AIME] Existing run candidate index: {self.config.evaluate_candidate_idx}")
        print(f"[AIME] Initial prompt actually passed to optimizer:\n{self.config.initial_prompt}")

    def evaluate(self, candidate: str, example) -> tuple[float, SideInfo]:
        score, side_info = self.task.evaluate_example(
            candidate,
            example,
            use_solver_cache=self.config.solver_cache_enabled,
        )
        if self.config.dataset_name == "mbpp":
            problem_id = side_info.get("problem_id", "?")
            passed = side_info.get("passed", False)
            feedback = str(side_info.get("execution_feedback", "")).strip()
            first_failure = ""
            failed_details = side_info.get("failed_details", []) or []
            if failed_details:
                item = failed_details[0]
                test = str(item.get("test", "")).strip()
                error = str(item.get("error", "")).strip()
                if test:
                    first_failure = f" first_failure_test={test!r}"
                if error:
                    first_failure += f" first_failure_error={error!r}"
            print(
                f"[MBPP][Eval] problem_id={problem_id} passed={passed} score={score:.2f}"
                f"{first_failure} feedback={feedback!r}"
            )
        return score, side_info

    def evaluate_prompt_on_batch(self, prompt: str, batch: list[Any]) -> BatchEvaluation:
        if not batch:
            return BatchEvaluation(prompt=prompt, example_evaluations=[])

        if self.config.parallel_evaluation and len(batch) > 1:
            with ThreadPoolExecutor(max_workers=min(self.config.max_workers, len(batch))) as executor:
                future_to_example = {executor.submit(self.evaluate, prompt, example): example for example in batch}
                results = []
                for future in as_completed(future_to_example):
                    example = future_to_example[future]
                    score, side_info = future.result()
                    results.append(ExampleEvaluation(example=example, score=score, side_info=side_info))
        else:
            results = [
                ExampleEvaluation(example=example, score=score, side_info=side_info)
                for example, (score, side_info) in (
                    (example, self.evaluate(prompt, example))
                    for example in batch
                )
            ]

        results.sort(key=lambda item: str(item.example.input))
        return BatchEvaluation(prompt=prompt, example_evaluations=results)

    def evaluate_prompt_summary(
        self,
        prompt: str,
        dataset: list[Any],
        *,
        pass_k: int = 1,
        cache_label: str | None = None,
    ) -> dict[str, Any]:
        return self.task.evaluate_dataset(
            prompt,
            dataset,
            max_workers=self.config.max_workers if self.config.parallel_evaluation else 1,
            use_solver_cache=self.config.eval_solver_cache_enabled,
            pass_k=pass_k,
            return_stats=True,
            cache_label=cache_label,
        )

    def _evaluate_optional_extra_testset(
        self,
        optimized_prompt: str,
        *,
        dataset_name: str,
        cache_path: Path,
        cache_label_prefix: str,
    ) -> dict[str, Any]:
        extra_testset = load_extra_math_testset(dataset_name, cache_path)
        print(f"\n[AIME] Evaluating optional extra test set: {dataset_name} ({len(extra_testset)} examples)")

        baseline_stats = self.evaluate_prompt_summary(
            self.config.initial_prompt,
            extra_testset,
            pass_k=self.config.eval_pass_k,
            cache_label=f"{cache_label_prefix}_baseline",
        )
        optimized_stats = self.evaluate_prompt_summary(
            optimized_prompt,
            extra_testset,
            pass_k=self.config.eval_pass_k,
            cache_label=f"{cache_label_prefix}_optimized",
        )
        print(
            f"[AIME] Extra test results for {dataset_name}: "
            f"baseline pass@{self.config.eval_pass_k}={baseline_stats['pass_score']:.2%}, "
            f"optimized pass@{self.config.eval_pass_k}={optimized_stats['pass_score']:.2%}, "
            f"delta={optimized_stats['pass_score'] - baseline_stats['pass_score']:.2%}"
        )
        return {
            "dataset_name": dataset_name,
            "num_examples": len(extra_testset),
            "baseline": baseline_stats,
            "optimized": optimized_stats,
        }

    def build_gepa_config(
        self,
        callbacks: list[object] | None = None,
        *,
        enable_merge: bool = False,
        max_merge_invocations: int = 5,
        merge_val_overlap_floor: int = 5,
    ) -> GEPAConfig:
        gepa_config = GEPAConfig(
            engine=EngineConfig(
                run_dir=str(self.config.run_dir),
                seed=self.config.seed,
                max_metric_calls=self.config.max_metric_calls,
                track_best_outputs=True,
                parallel=self.config.parallel_evaluation,
                max_workers=self.config.max_workers,
                num_parallel_proposals=self.config.num_parallel_proposals,
                cache_evaluation=False,
            ),
            reflection=ReflectionConfig(
                reflection_lm=self.reflection_lm,
                skip_perfect_score=True,
                perfect_score=1.0,
            ),
            merge=(
                MergeConfig(
                    max_merge_invocations=max_merge_invocations,
                    merge_val_overlap_floor=merge_val_overlap_floor,
                )
                if enable_merge
                else None
            ),
            callbacks=callbacks,
        )
        print(
            "[AIME] GEPA config: "
            f"run_dir={gepa_config.engine.run_dir}, "
            f"max_metric_calls={gepa_config.engine.max_metric_calls}, "
            f"parallel={gepa_config.engine.parallel}, "
            f"max_workers={gepa_config.engine.max_workers}, "
            f"num_parallel_proposals={gepa_config.engine.num_parallel_proposals}, "
            f"cache_evaluation={gepa_config.engine.cache_evaluation}, "
            f"skip_perfect_score={gepa_config.reflection.skip_perfect_score}, "
            f"perfect_score={gepa_config.reflection.perfect_score}, "
            f"merge_enabled={gepa_config.merge is not None}, "
            f"max_merge_invocations={getattr(gepa_config.merge, 'max_merge_invocations', 0)}, "
            f"merge_val_overlap_floor={getattr(gepa_config.merge, 'merge_val_overlap_floor', 0)}"
        )
        return gepa_config

    def select_optimized_prompt(self, result) -> tuple[str, int]:
        if self.config.best_candidate_strategy == "default":
            return str(result.best_candidate), int(result.best_idx)

        scores = list(result.val_aggregate_scores)
        if not scores:
            raise ValueError("GEPA result has no candidate scores.")

        best_score = max(scores)
        tied_indices = [idx for idx, score in enumerate(scores) if math.isclose(score, best_score)]
        depths = self._candidate_depths(result.parents)
        best_idx = max(tied_indices, key=lambda idx: (depths[idx], -idx))
        candidate = result.candidates[best_idx]
        prompt = candidate.get("current_candidate", str(candidate))
        return str(prompt), best_idx

    def load_existing_candidate_prompt(self, run_dir: Path, candidate_idx: int) -> tuple[str, float]:
        state = GEPAState.load(str(run_dir))
        if candidate_idx < 0 or candidate_idx >= len(state.program_candidates):
            raise IndexError(
                f"Candidate index {candidate_idx} is out of range for run {run_dir} "
                f"(num_candidates={len(state.program_candidates)})."
            )

        candidate = state.program_candidates[candidate_idx]
        prompt = candidate.get("current_candidate", str(candidate))
        val_score, _ = state.get_program_average_val_subset(candidate_idx)
        return str(prompt), float(val_score)

    def _candidate_depths(self, parents: list[list[int | None]]) -> list[int]:
        depths: list[int | None] = [None] * len(parents)

        def compute_depth(candidate_idx: int) -> int:
            cached = depths[candidate_idx]
            if cached is not None:
                return cached

            parent_ids = [parent for parent in parents[candidate_idx] if parent is not None]
            if not parent_ids:
                depths[candidate_idx] = 0
                return 0

            depth = 1 + max(compute_depth(parent_idx) for parent_idx in parent_ids)
            depths[candidate_idx] = depth
            return depth

        return [compute_depth(candidate_idx) for candidate_idx in range(len(parents))]

    def report_final_results(
        self,
        optimized_prompt: str,
        *,
        label: str,
        output_stem: str = "latest_run_result_plot",
        run_dir: Path | None = None,
    ) -> dict[str, Any]:
        if self.config.skip_baseline_eval:
            baseline_stats = {
                "pass_k": self.config.eval_pass_k,
                "pass_score": self.config.default_baseline_pass_score,
                "mean_score": self.config.default_baseline_mean_score,
                "total_examples": len(self.testset),
                "total_attempts": 0,
                "skipped": True,
            }
            print(
                "\n[AIME] Skipping baseline evaluation. "
                f"Using configured baseline scores: "
                f"pass@{self.config.eval_pass_k}={baseline_stats['pass_score']:.2%}, "
                f"mean@{self.config.eval_pass_k}={baseline_stats['mean_score']:.2%}"
            )
        else:
            print("\nEvaluating Baseline (Initial Prompt)...")
            baseline_stats = self.evaluate_prompt_summary(
                self.config.initial_prompt,
                self.testset,
                pass_k=self.config.eval_pass_k,
                cache_label=f"{self.config.backend}_{label}_baseline",
            )
            print(
                f"[AIME] Baseline evaluation finished: "
                f"pass@{self.config.eval_pass_k}={baseline_stats['pass_score']:.2%}, "
                f"mean@{self.config.eval_pass_k}={baseline_stats['mean_score']:.2%}"
            )

        print(f"\nEvaluating Best Optimized Program ({label})...")
        print(f"Best Prompt Found:\n{optimized_prompt}")
        optimized_stats = self.evaluate_prompt_summary(
            optimized_prompt,
            self.testset,
            pass_k=self.config.eval_pass_k,
            cache_label=f"{self.config.backend}_{label}_optimized",
        )
        print(
            f"[AIME] Optimized evaluation finished: "
            f"pass@{self.config.eval_pass_k}={optimized_stats['pass_score']:.2%}, "
            f"mean@{self.config.eval_pass_k}={optimized_stats['mean_score']:.2%}"
        )

        print(f"Baseline pass@{self.config.eval_pass_k} Score: {baseline_stats['pass_score']:.2%}")
        print(f"Optimized pass@{self.config.eval_pass_k} Score: {optimized_stats['pass_score']:.2%}")
        print(
            f"pass@{self.config.eval_pass_k} Improvement: "
            f"{optimized_stats['pass_score'] - baseline_stats['pass_score']:.2%}"
        )
        print(f"Baseline mean@{self.config.eval_pass_k} Score: {baseline_stats['mean_score']:.2%}")
        print(f"Optimized mean@{self.config.eval_pass_k} Score: {optimized_stats['mean_score']:.2%}")
        print(
            f"mean@{self.config.eval_pass_k} Improvement: "
            f"{optimized_stats['mean_score'] - baseline_stats['mean_score']:.2%}"
        )

        plot_summary = generate_evaluation_plot(
            run_dir=run_dir or self.config.run_dir,
            baseline_score=baseline_stats["pass_score"],
            optimized_score=optimized_stats["pass_score"],
            pass_k=self.config.eval_pass_k,
            baseline_mean_score=baseline_stats["mean_score"],
            optimized_mean_score=optimized_stats["mean_score"],
            output_stem=output_stem,
        )
        print(f"[AIME] Evaluation plot written to: {plot_summary['plot_path']}")
        if "plot_warning" in plot_summary:
            print(f"[AIME] Plot warning: {plot_summary['plot_warning']}")

        extra_test_results: dict[str, Any] = {}
        if self.config.dataset_name == "aime":
            if self.config.enable_hmmt_feb_2025_test:
                extra_test_results["hmmt_feb_2025"] = self._evaluate_optional_extra_testset(
                    optimized_prompt,
                    dataset_name=HMMT_FEB_2025_DATASET,
                    cache_path=HMMT_FEB_2025_TEST_JSONL,
                    cache_label_prefix=f"{self.config.backend}_{label}_hmmt_feb_2025",
                )
            if self.config.enable_hmmt_feb_2026_test:
                extra_test_results["hmmt_feb_2026"] = self._evaluate_optional_extra_testset(
                    optimized_prompt,
                    dataset_name=HMMT_FEB_2026_DATASET,
                    cache_path=HMMT_FEB_2026_TEST_JSONL,
                    cache_label_prefix=f"{self.config.backend}_{label}_hmmt_feb_2026",
                )

        return {
            "baseline": baseline_stats,
            "optimized": optimized_stats,
            "plot_summary": plot_summary,
            "extra_test_results": extra_test_results,
        }
