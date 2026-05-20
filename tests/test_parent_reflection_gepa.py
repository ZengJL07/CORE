import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.aime_math.parent_reflection_gepa.config import ParentReflectionGEPAConfig
from examples.aime_math.parent_reflection_gepa.runner import ParentReflectionGEPARunner


class DummyBatchEvaluation:
    def __init__(self, prompt: str, examples, score: float):
        self.prompt = prompt
        self.example_evaluations = [
            SimpleNamespace(
                example=example,
                score=score,
                side_info={
                    "input": example.input,
                    "reasoning": f"reasoning for {prompt}",
                    "output": f"output for {prompt}",
                    "execution_feedback": f"feedback for {prompt}",
                },
            )
            for example in examples
        ]

    @property
    def average_score(self) -> float:
        if not self.example_evaluations:
            return 0.0
        return sum(item.score for item in self.example_evaluations) / len(self.example_evaluations)

    @property
    def metric_calls(self) -> int:
        return len(self.example_evaluations)


class RecordingReflectionLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self.prompts = []
        self._lock = threading.Lock()

    def __call__(self, prompt: str) -> str:
        with self._lock:
            self.prompts.append(prompt)
            if not self._responses:
                raise AssertionError("No reflection response left for test.")
            return self._responses.pop(0)


class FakeExperiment:
    def __init__(self, *, reflection_lm, score_by_prompt, seed: int = 0):
        self.config = SimpleNamespace(
            seed=seed,
            max_workers=4,
            max_metric_calls=200,
            initial_prompt="initial prompt",
            run_dir=Path("/tmp/parent_reflection_gepa_test"),
        )
        self.reflection_lm = reflection_lm
        self.trainset = [
            SimpleNamespace(input="train-a"),
            SimpleNamespace(input="train-b"),
            SimpleNamespace(input="train-c"),
        ]
        self.valset = [
            SimpleNamespace(input="val-a"),
            SimpleNamespace(input="val-b"),
            SimpleNamespace(input="val-c"),
        ]
        self.score_by_prompt = score_by_prompt
        self.calls = []

    def evaluate_prompt_on_batch(self, prompt: str, batch):
        self.calls.append((prompt, tuple(example.input for example in batch)))
        return DummyBatchEvaluation(prompt, batch, self.score_by_prompt.get(prompt, 0.0))


def test_parent_reflection_config_allows_zero_probe_batch() -> None:
    config = ParentReflectionGEPAConfig(probe_val_batch_size=0)
    assert config.probe_val_batch_size == 0

    with pytest.raises(ValueError):
        ParentReflectionGEPAConfig(probe_val_batch_size=-1)


def test_runner_reuses_train_gate_when_probe_batch_is_disabled() -> None:
    reflection_lm = RecordingReflectionLM(
        [
            "<REFLECTION>focus on arithmetic</REFLECTION>"
            "<IMPROVED_PROMPT>improved prompt</IMPROVED_PROMPT>"
        ]
    )
    experiment = FakeExperiment(
        reflection_lm=reflection_lm,
        score_by_prompt={
            "initial prompt": 0.0,
            "improved prompt": 1.0,
        },
    )
    config = ParentReflectionGEPAConfig(
        candidate_pool_size=2,
        reflect_train_batch_size=1,
        probe_val_batch_size=0,
        max_outer_steps=1,
        num_parallel_branches=1,
        score_sampling_temperature=1.0,
    )

    runner = ParentReflectionGEPARunner(experiment, config)
    result = runner.run()

    assert result.best_prompt == "improved prompt"
    assert len(experiment.calls) == 4
    assert experiment.calls[0][0] == "initial prompt"
    assert experiment.calls[1][0] == "initial prompt"
    assert experiment.calls[2][0] == "improved prompt"
    assert experiment.calls[3][0] == "improved prompt"
    assert len(experiment.calls[0][1]) == 3
    assert len(experiment.calls[1][1]) == 1
    assert len(experiment.calls[2][1]) == 1
    assert len(experiment.calls[3][1]) == 3

    branch = result.outer_step_records[0].branch_records[0]
    assert branch.metadata["reused_train_gate_for_total"] is True
    assert branch.probe_val_ids == []

    child_candidates = [candidate for candidate in result.candidate_pool if candidate.prompt == "improved prompt"]
    assert len(child_candidates) == 1
    child = child_candidates[0]
    assert child.parent_prompt == "initial prompt"
    assert child.reflection_from_parent == "focus on arithmetic"


def test_runner_keeps_parallel_branch_structure_and_passes_parent_context() -> None:
    reflection_lm = RecordingReflectionLM(
        [
            "<REFLECTION>first reflection</REFLECTION>"
            "<IMPROVED_PROMPT>child one</IMPROVED_PROMPT>",
            "<REFLECTION>second reflection</REFLECTION>"
            "<IMPROVED_PROMPT>child two</IMPROVED_PROMPT>",
            "<REFLECTION>third reflection</REFLECTION>"
            "<IMPROVED_PROMPT>child two</IMPROVED_PROMPT>",
        ]
    )
    experiment = FakeExperiment(
        reflection_lm=reflection_lm,
        score_by_prompt={
            "initial prompt": 0.0,
            "child one": 0.7,
            "child two": 1.0,
        },
    )
    config = ParentReflectionGEPAConfig(
        candidate_pool_size=2,
        reflect_train_batch_size=1,
        probe_val_batch_size=0,
        max_outer_steps=2,
        num_parallel_branches=2,
        score_sampling_temperature=1.0,
    )

    runner = ParentReflectionGEPARunner(experiment, config)
    result = runner.run()

    assert len(result.outer_step_records) == 2
    assert len(result.outer_step_records[1].branch_records) == 2
    assert any("first reflection" in prompt and "initial prompt" in prompt for prompt in reflection_lm.prompts[1:])


def test_train_batch_sampler_prefers_known_failing_examples() -> None:
    reflection_lm = RecordingReflectionLM([])
    experiment = FakeExperiment(
        reflection_lm=reflection_lm,
        score_by_prompt={"initial prompt": 0.0},
        seed=0,
    )
    config = ParentReflectionGEPAConfig(
        reflect_train_batch_size=2,
        probe_val_batch_size=0,
        max_outer_steps=1,
        num_parallel_branches=1,
    )

    runner = ParentReflectionGEPARunner(experiment, config)
    current_candidate = runner._build_candidate_record(
        prompt=experiment.config.initial_prompt,
        parent_candidate=None,
        reflection_from_parent="",
    )
    current_candidate.train_scores_by_sample = {
        "train-a": 1.0,
        "train-b": 0.0,
        "train-c": 1.0,
    }

    _batch, batch_ids, batch_eval = runner._sample_train_batch_with_rejection(current_candidate)

    assert "train-b" in batch_ids
    assert batch_eval.average_score < 1.0
