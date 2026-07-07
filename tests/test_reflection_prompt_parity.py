"""Parity test: with parent history OFF, the GEPA aligned proposer and the
prompt_ucb runner must send byte-identical reflection prompts for the same
evaluation of the same candidate.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.aime_math.parent_reflection_gepa.gepa_aligned_proposer import (
    GEPAAlignedReflectionProposer,
)
from examples.aime_math.parent_reflection_gepa.reflection import (
    build_reflection_prompt,
    format_batch_feedback,
)


class _FakeExampleEval:
    def __init__(self, score, side_info):
        self.score = score
        self.side_info = side_info


class _FakeBatchEval:
    def __init__(self, evals):
        self.example_evaluations = evals


def _side_infos():
    return [
        {
            "score": 0.0,
            "input": "Find the number of ordered pairs (a, b).",
            "prompt": "SOLVER PROMPT TEXT",
            "output": "42",
            "reasoning": "Chain of reasoning one.",
            "execution_feedback": "Answer 42 is incorrect; expected 17.",
        },
        {
            "score": 1.0,
            "input": "Compute the remainder when N is divided by 1000.",
            "prompt": "SOLVER PROMPT TEXT",
            "output": "17",
            "reasoning": "Chain of reasoning two.",
            "execution_feedback": "Correct.",
        },
    ]


def _prompt_ucb_reflection_prompt(current_prompt, side_infos):
    evals = [_FakeExampleEval(si["score"], si) for si in side_infos]
    batch = _FakeBatchEval(evals)
    batch_score = sum(e.score for e in evals) / len(evals)
    return build_reflection_prompt(
        current_prompt=current_prompt,
        current_batch_feedback=format_batch_feedback(batch),
        current_batch_score=batch_score,
        parent_prompt=None,
        parent_reflection="",
        include_parent_history=False,
    )


def test_gepa_aligned_prompt_matches_prompt_ucb():
    current_prompt = "You are a careful AIME solver."
    side_infos = _side_infos()

    captured = {}

    def fake_reflection_lm(prompt):
        captured["prompt"] = prompt
        return "<REFLECTION>ok</REFLECTION>\n<IMPROVED_PROMPT>new prompt</IMPROVED_PROMPT>"

    proposer = GEPAAlignedReflectionProposer(fake_reflection_lm)
    # GEPA reflective_dataset: one component, records = raw side_info dicts.
    reflective_dataset = {"instruction": list(side_infos)}
    result = proposer({"instruction": current_prompt}, reflective_dataset, ["instruction"])

    gepa_prompt = captured["prompt"]
    ucb_prompt = _prompt_ucb_reflection_prompt(current_prompt, side_infos)

    assert gepa_prompt == ucb_prompt, (
        "Reflection prompts differ.\n"
        f"--- GEPA ---\n{gepa_prompt}\n--- prompt_ucb ---\n{ucb_prompt}"
    )
    assert result["instruction"] == "new prompt"


def test_parent_history_absent_from_aligned_prompt():
    """The aligned prompt must not contain any parent-history block."""
    side_infos = _side_infos()
    ucb_prompt = _prompt_ucb_reflection_prompt("p", side_infos)
    assert "parent parameter value" not in ucb_prompt.lower()
    assert "<REFLECTION>" in ucb_prompt
    assert "<IMPROVED_PROMPT>" in ucb_prompt
