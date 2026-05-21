from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_math_metric_accepts_integer_answers():
    dspy = pytest.importorskip("dspy")
    from examples.aime_math.utils import math_metric

    example = dspy.Example(input="1+1?", answer="2").with_inputs("input")
    prediction = dspy.Prediction(answer="2", reasoning="")

    score, feedback = math_metric(example, prediction)
    assert score == 1.0
    assert "correct" in feedback


def test_math_metric_accepts_latex_fraction_answers():
    dspy = pytest.importorskip("dspy")
    from examples.aime_math.utils import math_metric

    example = dspy.Example(input="fraction", answer="\\frac{1}{576}").with_inputs("input")
    prediction = dspy.Prediction(answer="1/576", reasoning="")

    score, feedback = math_metric(example, prediction)
    assert score == 1.0
    assert "correct" in feedback


def test_math_metric_accepts_factorial_expression_answers():
    dspy = pytest.importorskip("dspy")
    from examples.aime_math.utils import math_metric

    example = dspy.Example(input="factorial", answer="2^{25} \\cdot 26!").with_inputs("input")
    prediction = dspy.Prediction(answer="2**25 * factorial(26)", reasoning="")

    score, feedback = math_metric(example, prediction)
    assert score == 1.0
    assert "correct" in feedback


def test_math_metric_accepts_multi_value_answers():
    dspy = pytest.importorskip("dspy")
    from examples.aime_math.utils import math_metric

    answer = "\\frac{-1+\\sqrt{17}}{2}, \\frac{-1-\\sqrt{17}}{2}"
    example = dspy.Example(input="multi", answer=answer).with_inputs("input")
    prediction = dspy.Prediction(answer="(-1-sqrt(17))/2, (-1+sqrt(17))/2", reasoning="")

    score, feedback = math_metric(example, prediction)
    assert score == 1.0
    assert "correct" in feedback


def test_math_metric_rejects_wrong_expression_answer():
    dspy = pytest.importorskip("dspy")
    from examples.aime_math.utils import math_metric

    example = dspy.Example(input="wrong", answer="14+4\\sqrt{37}").with_inputs("input")
    prediction = dspy.Prediction(answer="14+4*sqrt(36)", reasoning="")

    score, feedback = math_metric(example, prediction)
    assert score == 0.0
    assert "incorrect" in feedback


def test_math_metric_accepts_repaired_partial_json_answer_fragment():
    dspy = pytest.importorskip("dspy")
    from examples.aime_math.utils import math_metric

    example = dspy.Example(input="fragment", answer="3375").with_inputs("input")
    prediction = dspy.Prediction(answer='": "3375"\n}\n```', reasoning="")

    score, feedback = math_metric(example, prediction)
    assert score == 1.0
    assert "correct" in feedback
