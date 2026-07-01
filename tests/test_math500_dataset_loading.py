from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _fake_math500_rows():
    return [
        {
            "problem": f"Problem {i}: compute something.",
            "answer": str(i) if i % 2 == 0 else r"\frac{1}{2}",
            "solution": f"Worked solution {i}.",
            "subject": "Algebra",
            "level": 3,
            "unique_id": f"test/algebra/{i}.json",
        }
        for i in range(10)
    ]


def test_math500_task_loads_and_splits(monkeypatch, tmp_path):
    dspy = pytest.importorskip("dspy")
    import examples.aime_math.utils as utils
    from examples.aime_math.utils import MATH500Task

    monkeypatch.setattr(utils, "MATH500_JSONL", tmp_path / "math500_test.jsonl")
    monkeypatch.setattr(utils, "DATA_DIR", tmp_path)

    def fake_load_dataset(name, split):
        assert name == "HuggingFaceH4/MATH-500"
        assert split == "test"
        return _fake_math500_rows()

    monkeypatch.setattr(utils, "load_dataset", fake_load_dataset)

    task = MATH500Task(train_size=5, val_size=3)
    splits = task.load_splits(seed=0)

    assert len(splits.trainset) == 5
    assert len(splits.valset) == 3
    assert len(splits.testset) == 2  # 10 - 5 - 3

    train_example = splits.trainset[0]
    assert isinstance(train_example, dspy.Example)
    assert train_example.input.startswith("Problem")
    # Train/val keep the worked solution for reflection; test drops it.
    assert getattr(train_example, "solution", "") != ""
    assert "solution" not in splits.testset[0].toDict()

    # The cache JSONL is written so subsequent loads avoid re-downloading.
    assert (tmp_path / "math500_test.jsonl").exists()


def test_math500_task_caches_and_reuses_local_jsonl(monkeypatch, tmp_path):
    pytest.importorskip("dspy")
    import examples.aime_math.utils as utils
    from examples.aime_math.utils import MATH500Task

    monkeypatch.setattr(utils, "MATH500_JSONL", tmp_path / "math500_test.jsonl")
    monkeypatch.setattr(utils, "DATA_DIR", tmp_path)

    call_count = {"n": 0}

    def fake_load_dataset(name, split):
        call_count["n"] += 1
        return _fake_math500_rows()

    monkeypatch.setattr(utils, "load_dataset", fake_load_dataset)

    task = MATH500Task(train_size=4, val_size=2)
    task.load_splits(seed=1)
    task.load_splits(seed=1)

    # Downloaded once; the second load reads the cached JSONL.
    assert call_count["n"] == 1


def test_math500_split_seed_is_deterministic(monkeypatch, tmp_path):
    pytest.importorskip("dspy")
    import examples.aime_math.utils as utils
    from examples.aime_math.utils import MATH500Task

    monkeypatch.setattr(utils, "MATH500_JSONL", tmp_path / "math500_test.jsonl")
    monkeypatch.setattr(utils, "DATA_DIR", tmp_path)
    monkeypatch.setattr(utils, "load_dataset", lambda name, split: _fake_math500_rows())

    task = MATH500Task(train_size=5, val_size=2)
    first = [ex.input for ex in task.load_splits(seed=7).trainset]
    second = [ex.input for ex in task.load_splits(seed=7).trainset]
    assert first == second


def test_math500_rejects_oversized_split(monkeypatch, tmp_path):
    pytest.importorskip("dspy")
    import examples.aime_math.utils as utils
    from examples.aime_math.utils import MATH500Task

    monkeypatch.setattr(utils, "MATH500_JSONL", tmp_path / "math500_test.jsonl")
    monkeypatch.setattr(utils, "DATA_DIR", tmp_path)
    monkeypatch.setattr(utils, "load_dataset", lambda name, split: _fake_math500_rows())

    task = MATH500Task(train_size=400, val_size=400)
    with pytest.raises(ValueError, match="exceeds dataset size"):
        task.load_splits(seed=0)


def test_build_dataset_task_returns_math500(monkeypatch):
    pytest.importorskip("dspy")
    from examples.aime_math.utils import MATH500Task, build_dataset_task

    task = build_dataset_task(
        "math500",
        math500_dataset="HuggingFaceH4/MATH-500",
        math500_train_size=200,
        math500_val_size=100,
    )
    assert isinstance(task, MATH500Task)
    assert task.train_size == 200
    assert task.val_size == 100


def test_solver_instructions_use_boxed_not_json():
    from examples.aime_math.utils import _build_solver_instructions

    instructions = _build_solver_instructions("Solve it.", include_reasoning=True, output_mode="integer")
    assert r"\boxed{" in instructions
    # The legacy JSON envelope must be gone for the math (integer) mode.
    assert "valid JSON object" not in instructions
    assert '{"answer"' not in instructions


def test_solver_instructions_python_mode_still_json():
    from examples.aime_math.utils import _build_solver_instructions

    instructions = _build_solver_instructions("Write code.", include_reasoning=True, output_mode="python_code")
    assert "valid JSON object" in instructions
    assert "```python" in instructions
