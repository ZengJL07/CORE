from __future__ import annotations

import json
import random
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from examples.aime_math.experiment import AIMEExperiment
from examples.aime_math.textgrad_original.config import OriginalTextGradConfig


def _ensure_textgrad_importable() -> None:
    if "diskcache" not in sys.modules:
        diskcache_stub = types.ModuleType("diskcache")

        class _Cache(dict):
            def __init__(self, _path: str | None = None):
                super().__init__()

            def close(self) -> None:
                return None

        diskcache_stub.Cache = _Cache
        sys.modules["diskcache"] = diskcache_stub

    if "platformdirs" not in sys.modules:
        platformdirs_stub = types.ModuleType("platformdirs")

        def _user_cache_dir(appname: str | None = None) -> str:
            base = Path("/tmp/textgrad_cache")
            if appname:
                base = base / appname
            base.mkdir(parents=True, exist_ok=True)
            return str(base)

        platformdirs_stub.user_cache_dir = _user_cache_dir
        sys.modules["platformdirs"] = platformdirs_stub

    textgrad_repo = "/home/jlzeng/code/textgrad"
    if textgrad_repo not in sys.path:
        sys.path.insert(0, textgrad_repo)


_ensure_textgrad_importable()

import textgrad as tg  # noqa: E402
from textgrad.engine import EngineLM  # noqa: E402
from textgrad.loss import MultiFieldEvaluation  # noqa: E402


@dataclass(frozen=True)
class OriginalTextGradRunResult:
    final_prompt: str
    final_val_score: float | None
    total_metric_calls: int
    total_steps: int
    prompt_history: list[dict[str, Any]]


class AIMEForwardEngine(EngineLM):
    model_string = "aime-forward-engine"
    system_prompt = "AIME solver engine"

    def __init__(self, experiment: AIMEExperiment):
        self.experiment = experiment

    def generate(self, prompt, system_prompt=None, **kwargs):
        return self.__call__(prompt, system_prompt=system_prompt, **kwargs)

    def __call__(self, prompt, system_prompt=None, **kwargs):
        example = getattr(prompt, "raw_example", None)
        if example is None:
            example = SimpleNamespace(input=str(prompt))
        answer, reasoning = self.experiment.task.generate_model_output(
            example,
            str(system_prompt or ""),
            use_solver_cache=self.experiment.config.solver_cache_enabled,
        )
        reasoning = str(reasoning).strip()
        answer = str(answer).strip()
        if reasoning and answer:
            return f"Reasoning:\n{reasoning}\n\nFinal answer:\n{answer}"
        if answer:
            return f"Final answer:\n{answer}"
        return reasoning


class AIMEBackwardEngine(EngineLM):
    model_string = "aime-backward-engine"
    system_prompt = "AIME backward engine"

    def __init__(self, experiment: AIMEExperiment):
        self.experiment = experiment

    def generate(self, prompt, system_prompt=None, **kwargs):
        return self.__call__(prompt, system_prompt=system_prompt, **kwargs)

    def __call__(self, prompt, system_prompt=None, **kwargs):
        rendered_prompt = str(prompt)
        if system_prompt:
            rendered_prompt = "\n".join(
                [
                    "<SYSTEM_PROMPT>",
                    str(system_prompt),
                    "</SYSTEM_PROMPT>",
                    "",
                    "<USER_PROMPT>",
                    str(prompt),
                    "</USER_PROMPT>",
                ]
            )
        return self.experiment.reflection_lm(rendered_prompt)


class OriginalTextGradRunner:
    def __init__(self, experiment: AIMEExperiment, config: OriginalTextGradConfig):
        self.experiment = experiment
        self.config = config
        self.rng = random.Random(experiment.config.seed)
        self.forward_engine = AIMEForwardEngine(experiment)
        self.backward_engine = AIMEBackwardEngine(experiment)
        tg.set_backward_engine(self.backward_engine, override=True)

    def _iter_train_batches(self):
        trainset = list(self.experiment.trainset)
        for _epoch in range(self.config.max_epochs):
            indices = list(range(len(trainset)))
            self.rng.shuffle(indices)
            for start in range(0, len(indices), self.config.batch_size):
                batch_indices = indices[start : start + self.config.batch_size]
                yield [trainset[idx] for idx in batch_indices]

    def _build_eval_fn(self):
        evaluation_instruction = tg.Variable(
            self.experiment.task.textgrad_evaluation_instruction(),
            requires_grad=False,
            role_description="evaluation instruction for AIME prompt optimization",
        )
        return MultiFieldEvaluation(
            evaluation_instruction,
            role_descriptions=[
                "Question for the task",
                "Ground truth answer",
                "Reasoning and prediction from the language model",
            ],
            engine=self.backward_engine,
        )

    def run(self) -> OriginalTextGradRunResult:
        print(
            "[AIME] Original TextGrad config: "
            f"algorithm={self.config.algorithm}, batch_size={self.config.batch_size}, "
            f"max_epochs={self.config.max_epochs}, max_steps={self.config.max_steps}, "
            "validation_frequency=1 (forced to match original prompt optimization loop), "
            f"revert_on_validation_drop={self.config.revert_on_validation_drop}"
        )

        system_prompt = tg.Variable(
            self.experiment.config.initial_prompt,
            requires_grad=True,
            role_description=(
                "structured system prompt to a somewhat capable language model that specifies behavior and strategies "
                "for solving AIME-style math problems"
            ),
        )
        model = tg.BlackboxLLM(self.forward_engine, system_prompt)
        eval_fn = self._build_eval_fn()
        optimizer = tg.TextualGradientDescent(engine=self.backward_engine, parameters=[system_prompt])

        prompt_history: list[dict[str, Any]] = []
        initial_prompt = system_prompt.get_value()
        initial_val_stats = self.experiment.evaluate_prompt_summary(
            initial_prompt,
            self.experiment.valset,
            pass_k=1,
            cache_label=f"{self.experiment.config.backend}_{self.config.algorithm}_initial_val",
        )
        initial_val_score = float(initial_val_stats["mean_score"])
        total_metric_calls = int(initial_val_stats["total_attempts"])
        previous_prompt = initial_prompt
        previous_val_score = initial_val_score
        final_val_score: float | None = initial_val_score
        print(f"[AIME] Initial validation score: {initial_val_score:.2%}")

        step_idx = 0
        valset_metric_cost = len(self.experiment.valset)
        for batch in self._iter_train_batches():
            if step_idx >= self.config.max_steps or total_metric_calls >= self.experiment.config.max_metric_calls:
                break

            remaining_budget = self.experiment.config.max_metric_calls - total_metric_calls
            if remaining_budget < len(batch):
                print(
                    f"[AIME] Stopping before step {step_idx + 1}: remaining budget={remaining_budget}, "
                    f"required_minimum={len(batch)}"
                )
                break

            print(
                f"[AIME] Original TextGrad step {step_idx + 1}: "
                f"batch_size={len(batch)}, prompt_chars={len(system_prompt.get_value())}"
            )
            optimizer.zero_grad()
            losses = []
            batch_predictions = []
            for example in batch:
                question_var = tg.Variable(
                    example.input,
                    requires_grad=False,
                    role_description="Question for the task",
                )
                question_var.raw_example = example
                answer_var = tg.Variable(
                    str(getattr(example, "canonical_solution", getattr(example, "answer", ""))),
                    requires_grad=False,
                    role_description="Ground truth answer",
                )
                response = model(question_var)
                batch_predictions.append(response.value)
                losses.append(eval_fn([question_var, answer_var, response]))

            total_metric_calls += len(batch)
            total_loss = tg.sum(losses)
            total_loss.backward()
            optimizer.step()

            val_score = None
            remaining_budget = self.experiment.config.max_metric_calls - total_metric_calls
            if remaining_budget >= valset_metric_cost:
                val_stats = self.experiment.evaluate_prompt_summary(
                    system_prompt.get_value(),
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
                    system_prompt.set_value(previous_prompt)
                    val_score = previous_val_score
                else:
                    previous_prompt = system_prompt.get_value()
                    previous_val_score = val_score
                final_val_score = val_score
            else:
                print(
                    f"[AIME] Skipping validation at step {step_idx + 1}: "
                    f"remaining budget={remaining_budget}, required={valset_metric_cost}"
                )

            prompt_history.append(
                {
                    "step_idx": step_idx,
                    "prompt": system_prompt.get_value(),
                    "val_score": val_score,
                    "batch_size": len(batch),
                    "sample_inputs": [example.input for example in batch],
                    "sample_predictions": batch_predictions,
                }
            )
            print(
                f"[AIME] Step {step_idx + 1} finished: "
                f"val_score={'n/a' if val_score is None else f'{val_score:.2%}'}, "
                f"metric_calls={total_metric_calls}"
            )
            step_idx += 1

        final_prompt = system_prompt.get_value()
        self._write_summary(
            prompt_history,
            final_prompt,
            final_val_score,
            initial_val_score,
            total_metric_calls,
            step_idx,
        )
        return OriginalTextGradRunResult(
            final_prompt=final_prompt,
            final_val_score=final_val_score,
            total_metric_calls=total_metric_calls,
            total_steps=step_idx,
            prompt_history=prompt_history,
        )

    def _write_summary(
        self,
        prompt_history: list[dict[str, Any]],
        final_prompt: str,
        final_val_score: float | None,
        initial_val_score: float,
        total_metric_calls: int,
        total_steps: int,
    ) -> None:
        summary = {
            "algorithm": self.config.algorithm,
            "implementation": "original_textgrad",
            "final_prompt": final_prompt,
            "final_val_score": final_val_score,
            "initial_val_score": initial_val_score,
            "total_metric_calls": total_metric_calls,
            "total_steps": total_steps,
            "prompt_history": prompt_history,
        }
        Path(self.experiment.config.run_dir).mkdir(parents=True, exist_ok=True)
        summary_path = Path(self.experiment.config.run_dir) / f"textgrad_{self.config.algorithm}_summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
