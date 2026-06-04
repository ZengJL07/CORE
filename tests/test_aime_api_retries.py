import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _no_sleep(monkeypatch):
    # Keep the retry/backoff logic exercised but instantaneous in tests.
    monkeypatch.setattr("examples.aime_math.utils.time.sleep", lambda _seconds: None)


def test_call_with_transient_retries_recovers_after_failures(monkeypatch):
    from examples.aime_math.utils import call_with_transient_retries

    _no_sleep(monkeypatch)
    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ConnectionError("transient boom")
        return "ok"

    result = call_with_transient_retries(flaky, max_retries=5, description="unit call")
    assert result == "ok"
    assert attempts["n"] == 3


def test_call_with_transient_retries_reraises_after_exhaustion(monkeypatch):
    from examples.aime_math.utils import call_with_transient_retries

    _no_sleep(monkeypatch)
    attempts = {"n": 0}

    def always_fails():
        attempts["n"] += 1
        raise ConnectionError("still broken")

    with pytest.raises(ConnectionError):
        call_with_transient_retries(always_fails, max_retries=4, description="unit call")
    assert attempts["n"] == 4


def test_call_with_transient_retries_does_not_retry_parse_errors(monkeypatch):
    import dspy
    from dspy.utils.exceptions import AdapterParseError

    from examples.aime_math.utils import call_with_transient_retries

    _no_sleep(monkeypatch)
    attempts = {"n": 0}

    class _Sig(dspy.Signature):
        question = dspy.InputField()
        answer = dspy.OutputField()

    def parse_failure():
        attempts["n"] += 1
        raise AdapterParseError(
            adapter_name="test",
            signature=_Sig,
            lm_response="not json",
        )

    with pytest.raises(AdapterParseError):
        call_with_transient_retries(parse_failure, max_retries=5, description="unit call")
    # Parse errors are surfaced immediately so the caller's recovery logic runs;
    # they are not treated as transient and must not be retried.
    assert attempts["n"] == 1


def test_solver_retries_transient_failures_then_succeeds(monkeypatch, tmp_path):
    dspy = pytest.importorskip("dspy")
    from examples.aime_math.utils import MathSolverClient, evaluate_on_dataset

    _no_sleep(monkeypatch)

    class FlakySolverClient(MathSolverClient):
        def __init__(self, cache_dir: Path):
            super().__init__(cache_dir=cache_dir, enable_cache=False, api_max_retries=5)
            self.calls = 0

        def _call_solver(self, example, instructions: str, include_reasoning: bool):
            self.calls += 1
            if self.calls < 3:
                raise ConnectionError("transient connection reset")
            return dspy.Prediction(answer=str(example.answer), reasoning=f"call {self.calls}")

    client = FlakySolverClient(tmp_path)
    monkeypatch.setattr("examples.aime_math.utils._DEFAULT_SOLVER_CLIENT", client)
    dataset = [dspy.Example(input="1+1?", answer="2").with_inputs("input")]

    # Without retries this would collapse to an empty prediction (score 0.0); with
    # api_max_retries=5 the third attempt succeeds and the example is scored 1.0.
    assert evaluate_on_dataset("solve", dataset, max_workers=1, use_solver_cache=False) == 1.0
    assert client.calls == 3


def test_solver_returns_empty_after_exhausting_retries(monkeypatch, tmp_path):
    dspy = pytest.importorskip("dspy")
    from examples.aime_math.utils import MathSolverClient

    _no_sleep(monkeypatch)

    class AlwaysFailingSolverClient(MathSolverClient):
        def __init__(self, cache_dir: Path):
            super().__init__(cache_dir=cache_dir, enable_cache=True, api_max_retries=3)
            self.calls = 0

        def _call_solver(self, example, instructions: str, include_reasoning: bool):
            self.calls += 1
            raise ConnectionError("permanent outage")

    client = AlwaysFailingSolverClient(tmp_path)
    example = dspy.Example(input="1+1?", answer="2").with_inputs("input")

    prediction = client.predict(example, "solve")
    assert prediction.answer == ""
    # All api_max_retries attempts were spent before giving up.
    assert client.calls == 3
    # Transient failures must NOT poison the cache with an empty answer.
    assert list(tmp_path.glob("*.json")) == []


def test_mbpp_pass_k_uses_distinct_cache_keys_per_attempt(monkeypatch, tmp_path):
    dspy = pytest.importorskip("dspy")
    from examples.aime_math.utils import MathSolverClient, MBPPTask

    canonical = "def add_one(x):\n    return x + 1\n"
    example = dspy.Example(
        input="Write a function add_one.",
        answer=canonical,
        problem_id=1,
        content="Write a function add_one.",
        function_name="add_one",
        signature_line="def add_one(x):",
        canonical_solution=canonical,
        test_list=["assert add_one(1) == 2"],
        challenge_test_list=[],
        test_setup_code="",
    ).with_inputs("input")

    # First attempt returns a wrong solution; later attempts return the correct one.
    # If pass@k attempts shared one cache key, the wrong first answer would be reused
    # for every attempt (pass@k would collapse to pass@1 => pass_score 0.0).
    code_by_call = {
        0: "```python\ndef add_one(x):\n    return x - 1\n```",
        1: f"```python\n{canonical}```",
        2: f"```python\n{canonical}```",
    }

    class SequencedSolverClient(MathSolverClient):
        def __init__(self, cache_dir: Path):
            super().__init__(
                cache_dir=cache_dir,
                enable_cache=True,
                output_mode="python_code",
                api_max_retries=5,
            )
            self.calls = 0

        def _call_solver(self, example, instructions: str, include_reasoning: bool):
            answer = code_by_call[min(self.calls, 2)]
            self.calls += 1
            return dspy.Prediction(answer=answer, reasoning="")

    client = SequencedSolverClient(tmp_path)
    monkeypatch.setattr("examples.aime_math.utils._DEFAULT_SOLVER_CLIENT", client)

    task = MBPPTask(source="huggingface")
    stats = task.evaluate_dataset(
        "solve",
        [example],
        max_workers=1,
        use_solver_cache=True,
        pass_k=3,
        return_stats=True,
    )
    # Each attempt got its own cache key, so the solver was actually invoked 3 times
    # and at least one correct attempt yields pass_score 1.0.
    assert client.calls == 3
    assert stats["pass_score"] == 1.0
