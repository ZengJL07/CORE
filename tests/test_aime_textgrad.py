import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.aime_math.prompt_optimizers.optimizer import (
    TextGradAlgorithmConfig,
    TextualGradientDescentMomentumOptimizer,
    TextualGradientDescentOptimizer,
    build_prompt_optimizer,
)
from examples.aime_math.prompt_optimizers.autograd import TextualLoss
from examples.aime_math.prompt_optimizers.sampling import (
    gumbel_topk_indices,
    softmax,
    weighted_sample_without_replacement,
)
from examples.aime_math.prompt_optimizers.state import PromptVariable


class DummyEngine:
    def __call__(self, prompt: str) -> str:
        return prompt


class PromptUpdateEngine:
    def __call__(self, prompt: str) -> str:
        return "<IMPROVED_PROMPT>improved prompt</IMPROVED_PROMPT>"


class GradientEngine:
    def __call__(self, prompt: str) -> str:
        return "<TEXT_GRADIENT>focus on arithmetic mistakes</TEXT_GRADIENT>"


class DummyBatchEvaluation:
    metric_calls = 2
    average_score = 0.5


def make_config(algorithm: str) -> TextGradAlgorithmConfig:
    return TextGradAlgorithmConfig(
        algorithm=algorithm,
        batch_size=2,
        max_epochs=1,
        max_steps=2,
        candidate_pool_size=3,
        top_k=2,
        gumbel_temperature=1.0,
        momentum_buffer_size=3,
        momentum_size=2,
        bootstrap_samples=2,
        validation_frequency=1,
        revert_on_validation_drop=False,
    )


def test_prompt_variable_collects_gradients() -> None:
    variable = PromptVariable(value="prompt", role_description="system prompt")
    variable.add_gradient(
        "fix arithmetic",
        0.0,
        context={
            "problem_input": "1+1",
            "model_output": "3",
            "model_reasoning": "bad math",
            "response_desc": "the final answer",
            "variable_desc": "system prompt",
        },
    )
    variable.add_gradient("be stricter about final answer formatting", 1.0)

    gradient_text = variable.get_gradient_text()
    gradient_and_context = variable.get_gradient_and_context_text()

    assert "fix arithmetic" in gradient_text
    assert "final answer formatting" in gradient_text
    assert "<CONVERSATION>" in gradient_and_context
    assert "<FEEDBACK>fix arithmetic</FEEDBACK>" in gradient_and_context


def test_textual_loss_backward_populates_variable_gradients() -> None:
    variable = PromptVariable(value="prompt", role_description="system prompt")
    loss = TextualLoss(
        prompt_variable=variable,
        problem_input="1+1",
        model_output="3",
        model_reasoning="I guessed",
        score=0.0,
        feedback="The correct answer is 2.",
    )

    loss.backward(GradientEngine())

    assert variable.get_gradient_text() == "focus on arithmetic mistakes"


def test_tgd_step_consumes_existing_gradients() -> None:
    variable = PromptVariable(value="prompt", role_description="system prompt")
    variable.add_gradient("fix arithmetic", 0.0)
    optimizer = TextualGradientDescentOptimizer(
        variable=variable,
        engine=PromptUpdateEngine(),
        config=make_config("tgd"),
        rng=random.Random(0),
    )

    class StubExperiment:
        def evaluate_prompt_on_batch(self, prompt, batch):
            assert prompt == "improved prompt"
            return DummyBatchEvaluation()

    result = optimizer.step(
        experiment=StubExperiment(),
        batch=["example"],
        current_batch_evaluation=DummyBatchEvaluation(),
        step_idx=0,
        prompt_history=[],
    )

    assert variable.value == "improved prompt"
    assert result.prompt == "improved prompt"
    assert "fix arithmetic" in result.combined_gradient


def test_softmax_normalizes_probabilities() -> None:
    probs = softmax([1.0, 2.0, 3.0])
    assert pytest.approx(sum(probs), rel=1e-6, abs=1e-6) == 1.0
    assert probs[2] > probs[1] > probs[0]


def test_gumbel_topk_indices_returns_unique_indices() -> None:
    rng = random.Random(0)
    indices = gumbel_topk_indices([0.1, 0.5, 0.3, 0.9], 2, tau=1.0, rng=rng)

    assert len(indices) == 2
    assert len(set(indices)) == 2
    assert all(idx in {0, 1, 2, 3} for idx in indices)


def test_weighted_sample_without_replacement_is_unique() -> None:
    rng = random.Random(1)
    sampled = weighted_sample_without_replacement([0, 1, 2, 3], [0.1, 0.2, 0.3, 0.4], 3, rng=rng)

    assert len(sampled) == 3
    assert len(set(sampled)) == 3


def test_build_prompt_optimizer_supports_tgd_and_tsgd_m() -> None:
    variable = PromptVariable(value="prompt", role_description="system prompt")

    tgd = build_prompt_optimizer(
        variable=variable,
        engine=DummyEngine(),
        config=make_config("tgd"),
        rng=random.Random(0),
    )
    tsgd_m = build_prompt_optimizer(
        variable=variable,
        engine=DummyEngine(),
        config=make_config("tsgd_m"),
        rng=random.Random(0),
    )

    assert isinstance(tgd, TextualGradientDescentOptimizer)
    assert isinstance(tsgd_m, TextualGradientDescentMomentumOptimizer)
