from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_solver_cache_can_be_disabled_per_evaluation(monkeypatch, tmp_path):
    dspy = pytest.importorskip("dspy")
    from examples.aime_math.utils import MathSolverClient, evaluate_on_dataset

    class CountingSolverClient(MathSolverClient):
        def __init__(self, cache_dir: Path):
            super().__init__(cache_dir=cache_dir, enable_cache=True)
            self.calls = 0

        def _call_solver(self, example, instructions: str, include_reasoning: bool):
            self.calls += 1
            return dspy.Prediction(answer=str(example.answer), reasoning=f"call {self.calls}")

    client = CountingSolverClient(tmp_path)
    monkeypatch.setattr("examples.aime_math.utils._DEFAULT_SOLVER_CLIENT", client)
    dataset = [dspy.Example(input="1+1?", answer="2").with_inputs("input")]

    assert evaluate_on_dataset("solve", dataset, max_workers=1) == 1.0
    assert client.calls == 1
    assert len(list(tmp_path.glob("*.json"))) == 1

    assert evaluate_on_dataset("solve", dataset, max_workers=1) == 1.0
    assert client.calls == 1

    assert evaluate_on_dataset("solve", dataset, max_workers=1, use_solver_cache=False) == 1.0
    assert client.calls == 2

    uncached_dir_count = len(list(tmp_path.glob("*.json")))
    assert evaluate_on_dataset("new prompt", dataset, max_workers=1, use_solver_cache=False) == 1.0
    assert client.calls == 3
    assert len(list(tmp_path.glob("*.json"))) == uncached_dir_count


def test_pass_k_counts_any_success(monkeypatch, tmp_path):
    dspy = pytest.importorskip("dspy")
    from examples.aime_math.utils import MathSolverClient, evaluate_on_dataset

    class EventuallyCorrectSolverClient(MathSolverClient):
        def __init__(self, cache_dir: Path):
            super().__init__(cache_dir=cache_dir, enable_cache=False)
            self.calls = 0

        def _call_solver(self, example, instructions: str, include_reasoning: bool):
            self.calls += 1
            answer = "0" if self.calls == 1 else str(example.answer)
            return dspy.Prediction(answer=answer, reasoning=f"call {self.calls}")

    client = EventuallyCorrectSolverClient(tmp_path)
    monkeypatch.setattr("examples.aime_math.utils._DEFAULT_SOLVER_CLIENT", client)
    dataset = [dspy.Example(input="1+1?", answer="2").with_inputs("input")]

    stats = evaluate_on_dataset(
        "solve",
        dataset,
        max_workers=1,
        use_solver_cache=False,
        pass_k=3,
        return_stats=True,
    )
    assert stats["pass_score"] == 1.0
    assert stats["mean_score"] == 2 / 3
    assert client.calls == 3


def test_pass_k_one_with_cache_label_reuses_base_solver_cache(monkeypatch, tmp_path):
    dspy = pytest.importorskip("dspy")
    from examples.aime_math.utils import MathSolverClient, evaluate_on_dataset

    class CountingSolverClient(MathSolverClient):
        def __init__(self, cache_dir: Path):
            super().__init__(cache_dir=cache_dir, enable_cache=True)
            self.calls = 0

        def _call_solver(self, example, instructions: str, include_reasoning: bool):
            self.calls += 1
            return dspy.Prediction(answer=str(example.answer), reasoning=f"call {self.calls}")

    client = CountingSolverClient(tmp_path)
    monkeypatch.setattr("examples.aime_math.utils._DEFAULT_SOLVER_CLIENT", client)
    dataset = [dspy.Example(input="1+1?", answer="2").with_inputs("input")]

    base_stats = evaluate_on_dataset("solve", dataset, max_workers=1, pass_k=1, return_stats=True)
    assert base_stats["pass_score"] == 1.0
    assert client.calls == 1
    assert len(list(tmp_path.glob("*.json"))) == 1

    labeled_stats = evaluate_on_dataset(
        "solve",
        dataset,
        max_workers=1,
        pass_k=1,
        cache_label="textgrad_original_tgd_initial_val",
        return_stats=True,
    )
    assert labeled_stats["pass_score"] == 1.0
    assert client.calls == 1
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_pass_k_many_reuses_cache_across_cache_labels(monkeypatch, tmp_path):
    dspy = pytest.importorskip("dspy")
    from examples.aime_math.utils import MathSolverClient, evaluate_on_dataset

    class CountingSolverClient(MathSolverClient):
        def __init__(self, cache_dir: Path):
            super().__init__(cache_dir=cache_dir, enable_cache=True)
            self.calls = 0

        def _call_solver(self, example, instructions: str, include_reasoning: bool):
            self.calls += 1
            return dspy.Prediction(answer=str(example.answer), reasoning=f"call {self.calls}")

    client = CountingSolverClient(tmp_path)
    monkeypatch.setattr("examples.aime_math.utils._DEFAULT_SOLVER_CLIENT", client)
    dataset = [dspy.Example(input="1+1?", answer="2").with_inputs("input")]

    baseline_stats = evaluate_on_dataset(
        "solve",
        dataset,
        max_workers=1,
        pass_k=3,
        cache_label="trace2skill_baseline_baseline",
        return_stats=True,
    )
    assert baseline_stats["pass_score"] == 1.0
    assert client.calls == 3
    assert len(list(tmp_path.glob("*.json"))) == 3

    labeled_stats = evaluate_on_dataset(
        "solve",
        dataset,
        max_workers=1,
        pass_k=3,
        cache_label="textgrad_tgd_baseline",
        return_stats=True,
    )
    assert labeled_stats["pass_score"] == 1.0
    assert client.calls == 3
    assert len(list(tmp_path.glob("*.json"))) == 3


def test_pass_k_many_can_read_legacy_phase_labeled_cache(monkeypatch, tmp_path):
    dspy = pytest.importorskip("dspy")
    from examples.aime_math.utils import MathSolverClient, _prediction_cache_key, evaluate_on_dataset

    class CountingSolverClient(MathSolverClient):
        def __init__(self, cache_dir: Path):
            super().__init__(cache_dir=cache_dir, enable_cache=True)
            self.calls = 0

        def _call_solver(self, example, instructions: str, include_reasoning: bool):
            self.calls += 1
            return dspy.Prediction(answer=str(example.answer), reasoning=f"call {self.calls}")

    client = CountingSolverClient(tmp_path)
    monkeypatch.setattr("examples.aime_math.utils._DEFAULT_SOLVER_CLIENT", client)
    dataset = [dspy.Example(input="1+1?", answer="2").with_inputs("input")]

    problem_input = str(dataset[0].input)
    prompt = "solve"
    for attempt_idx in range(3):
        legacy_cache_extra = {
            "phase": "trace2skill_baseline_trace2skill_baseline_baseline",
            "pass_k": 3,
            "attempt_idx": attempt_idx,
        }
        legacy_path = tmp_path / f"{_prediction_cache_key(problem_input, prompt, None, legacy_cache_extra)}.json"
        legacy_path.write_text(
            (
                "{\n"
                f'  "input": {problem_input!r},\n'
                f'  "prompt": {prompt!r},\n'
                f'  "answer": "2",\n'
                f'  "reasoning": "legacy",\n'
                '  "source": "legacy",\n'
                f'  "cache_extra": {legacy_cache_extra!r}\n'
                "}\n"
            ).replace("'", '"'),
            encoding="utf-8",
        )

    # Simulate a fresh process loading only from the old phase-labeled files.
    fresh_client = CountingSolverClient(tmp_path)
    monkeypatch.setattr("examples.aime_math.utils._DEFAULT_SOLVER_CLIENT", fresh_client)

    reused_stats = evaluate_on_dataset(
        "solve",
        dataset,
        max_workers=1,
        pass_k=3,
        cache_label="textgrad_tgd_baseline",
        return_stats=True,
    )
    assert reused_stats["pass_score"] == 1.0
    assert fresh_client.calls == 0


def test_pass_k_many_can_read_legacy_phase_cache_from_other_root(monkeypatch, tmp_path):
    dspy = pytest.importorskip("dspy")
    from examples.aime_math.utils import MathSolverClient, _prediction_cache_key, evaluate_on_dataset

    prompt_ucb_root = tmp_path / "outputs" / "aime_math" / "prompt_UCB" / "api_cache" / "seed_42"
    textgrad_root = tmp_path / "outputs" / "aime_math" / "textgrad" / "api_cache" / "seed_42"
    prompt_ucb_root.mkdir(parents=True, exist_ok=True)
    textgrad_root.mkdir(parents=True, exist_ok=True)

    class CountingSolverClient(MathSolverClient):
        def __init__(self, cache_dir: Path):
            super().__init__(
                cache_dir=cache_dir,
                cache_namespace={
                    "solver_backend": "litellm_deepseek_json_v1",
                    "model": "openai/deepseek-chat",
                    "temperature": 0.2,
                    "max_tokens": 32000,
                    "backend": "parent_reflection_gepa",
                },
                enable_cache=True,
            )
            self.calls = 0

        def _call_solver(self, example, instructions: str, include_reasoning: bool):
            self.calls += 1
            return dspy.Prediction(answer=str(example.answer), reasoning=f"call {self.calls}")

    problem_input = "1+1?"
    prompt = "solve"
    legacy_namespace = {
        "solver_backend": "litellm_deepseek_json_v1",
        "model": "openai/deepseek-chat",
        "api_base": "https://api.deepseek.com/v1",
        "temperature": 0.2,
        "max_tokens": 32000,
        "backend": "textgrad",
    }
    for attempt_idx in range(3):
        legacy_cache_extra = {
            "phase": "textgrad_tgd_baseline",
            "pass_k": 3,
            "attempt_idx": attempt_idx,
        }
        legacy_path = textgrad_root / f"{_prediction_cache_key(problem_input, prompt, legacy_namespace, legacy_cache_extra)}.json"
        legacy_path.write_text(
            json.dumps(
                {
                    "input": problem_input,
                    "prompt": prompt,
                    "answer": "2",
                    "reasoning": "legacy",
                    "source": "legacy",
                    "namespace": legacy_namespace,
                    "cache_extra": legacy_cache_extra,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    monkeypatch.chdir(tmp_path)
    client = CountingSolverClient(prompt_ucb_root)
    monkeypatch.setattr("examples.aime_math.utils._DEFAULT_SOLVER_CLIENT", client)
    dataset = [dspy.Example(input=problem_input, answer="2").with_inputs("input")]

    reused_stats = evaluate_on_dataset(
        prompt,
        dataset,
        max_workers=1,
        pass_k=3,
        cache_label="trace2skill_baseline_baseline",
        return_stats=True,
    )
    assert reused_stats["pass_score"] == 1.0
    assert client.calls == 0


def test_base_solver_cache_can_read_legacy_cache_from_other_root(monkeypatch, tmp_path):
    dspy = pytest.importorskip("dspy")
    from examples.aime_math.utils import MathSolverClient, _prediction_cache_key, run_llm

    prompt_ucb_root = tmp_path / "outputs" / "aime_math" / "prompt_UCB" / "api_cache" / "seed_42"
    textgrad_root = tmp_path / "outputs" / "aime_math" / "textgrad" / "api_cache" / "seed_42"
    prompt_ucb_root.mkdir(parents=True, exist_ok=True)
    textgrad_root.mkdir(parents=True, exist_ok=True)

    class CountingSolverClient(MathSolverClient):
        def __init__(self, cache_dir: Path):
            super().__init__(
                cache_dir=cache_dir,
                cache_namespace={
                    "solver_backend": "litellm_deepseek_json_v1",
                    "model": "openai/deepseek-chat",
                    "temperature": 0.2,
                    "max_tokens": 32000,
                    "backend": "parent_reflection_gepa",
                },
                enable_cache=True,
            )
            self.calls = 0

        def _call_solver(self, example, instructions: str, include_reasoning: bool):
            self.calls += 1
            return dspy.Prediction(answer=str(example.answer), reasoning=f"call {self.calls}")

    problem_input = "1+1?"
    prompt = "solve"
    legacy_namespace = {
        "solver_backend": "litellm_deepseek_json_v1",
        "model": "openai/deepseek-chat",
        "api_base": "https://api.deepseek.com/v1",
        "temperature": 0.2,
        "max_tokens": 32000,
        "backend": "textgrad",
    }
    legacy_path = textgrad_root / f"{_prediction_cache_key(problem_input, prompt, legacy_namespace, None)}.json"
    legacy_path.write_text(
        json.dumps(
            {
                "input": problem_input,
                "prompt": prompt,
                "answer": "2",
                "reasoning": "legacy",
                "source": "legacy",
                "namespace": legacy_namespace,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    client = CountingSolverClient(prompt_ucb_root)
    monkeypatch.setattr("examples.aime_math.utils._DEFAULT_SOLVER_CLIENT", client)
    example = dspy.Example(input=problem_input, answer="2").with_inputs("input")

    prediction = run_llm(example, prompt, use_solver_cache=True)
    assert prediction.answer == "2"
    assert client.calls == 0


def test_solver_cache_namespace_backend_can_be_overridden(monkeypatch, tmp_path):
    from examples.aime_math.config import AIMEExperimentConfig

    monkeypatch.setenv("AIME_OUTPUT_ROOT", str(tmp_path))
    monkeypatch.setenv("AIME_SOLVER_CACHE_NAMESPACE_BACKEND", "parent_reflection_gepa")

    config = AIMEExperimentConfig.from_env("gepa")

    assert config.backend == "gepa"
    assert config.solver_cache_namespace()["backend"] == "parent_reflection_gepa"


def test_base_solver_cache_hit_skips_cross_root_index_build(monkeypatch, tmp_path):
    dspy = pytest.importorskip("dspy")
    from examples.aime_math.utils import MathSolverClient, _prediction_cache_key, run_llm

    cache_namespace = {
        "solver_backend": "litellm_deepseek_json_v1",
        "model": "openai/deepseek-chat",
        "temperature": 0.2,
        "max_tokens": 32000,
        "backend": "parent_reflection_gepa",
    }

    problem_input = "1+1?"
    prompt = "solve"
    cache_path = tmp_path / f"{_prediction_cache_key(problem_input, prompt, cache_namespace, None)}.json"
    cache_path.write_text(
        (
            "{\n"
            f'  "input": {problem_input!r},\n'
            f'  "prompt": {prompt!r},\n'
            '  "answer": "2",\n'
            '  "reasoning": "cached",\n'
            '  "source": "test",\n'
            f'  "namespace": {cache_namespace!r}\n'
            "}\n"
        ).replace("'", '"'),
        encoding="utf-8",
    )

    class GuardedSolverClient(MathSolverClient):
        def __init__(self, cache_dir: Path):
            super().__init__(cache_dir=cache_dir, cache_namespace=cache_namespace, enable_cache=True)
            self.calls = 0

        def _call_solver(self, example, instructions: str, include_reasoning: bool):
            self.calls += 1
            return dspy.Prediction(answer=str(example.answer), reasoning="live")

        def _get_cross_root_exact_index(self):
            raise AssertionError("cross-root exact index should not be built for direct cache hits")

        def _get_cross_root_phase_agnostic_index(self):
            raise AssertionError("cross-root phase index should not be built for direct cache hits")

    client = GuardedSolverClient(tmp_path)
    monkeypatch.setattr("examples.aime_math.utils._DEFAULT_SOLVER_CLIENT", client)
    example = dspy.Example(input=problem_input, answer="2").with_inputs("input")

    prediction = run_llm(example, prompt, use_solver_cache=True)
    assert prediction.answer == "2"
    assert client.calls == 0


def test_cached_prediction_is_reextracted_on_cache_hit(monkeypatch, tmp_path):
    dspy = pytest.importorskip("dspy")
    from examples.aime_math.utils import MathSolverClient, _prediction_cache_key, run_llm

    cache_namespace = {
        "solver_backend": "litellm_deepseek_json_v1",
        "model": "openai/deepseek-chat",
        "temperature": 0.2,
        "max_tokens": 32000,
        "backend": "parent_reflection_gepa",
    }

    problem_input = "cached broken answer"
    prompt = "solve"
    cache_path = tmp_path / f"{_prediction_cache_key(problem_input, prompt, cache_namespace, None)}.json"
    cache_path.write_text(
        (
            "{\n"
            f'  "input": {problem_input!r},\n'
            f'  "prompt": {prompt!r},\n'
            '  "answer": "\\": \\"3375\\"\\n}\\n```",\n'
            '  "reasoning": "cached",\n'
            '  "source": "test",\n'
            f'  "namespace": {cache_namespace!r}\n'
            "}\n"
        ).replace("'", '"'),
        encoding="utf-8",
    )

    class GuardedSolverClient(MathSolverClient):
        def __init__(self, cache_dir: Path):
            super().__init__(cache_dir=cache_dir, cache_namespace=cache_namespace, enable_cache=True)
            self.calls = 0

        def _call_solver(self, example, instructions: str, include_reasoning: bool):
            self.calls += 1
            return dspy.Prediction(answer=str(example.answer), reasoning="live")

    client = GuardedSolverClient(tmp_path)
    monkeypatch.setattr("examples.aime_math.utils._DEFAULT_SOLVER_CLIENT", client)
    example = dspy.Example(input=problem_input, answer="3375").with_inputs("input")

    prediction = run_llm(example, prompt, use_solver_cache=True)
    assert prediction.answer == "3375"
    assert client.calls == 0
