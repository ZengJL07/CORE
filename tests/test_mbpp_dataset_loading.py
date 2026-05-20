from pathlib import Path

import pytest


def test_mbpp_task_loads_huggingface_schema(monkeypatch):
    dspy = pytest.importorskip("dspy")
    from examples.aime_math.utils import MBPPTask

    rows = {
        "train": [
            {
                "task_id": 1,
                "text": "Write a function to add two integers.",
                "code": "def add_numbers(a, b):\n    return a + b",
                "test_list": ["assert add_numbers(1, 2) == 3"],
                "test_setup_code": "",
                "challenge_test_list": ["assert add_numbers(-1, 1) == 0"],
            }
        ],
        "validation": [
            {
                "task_id": 2,
                "text": "Write a function to square a number.",
                "code": "def square(x):\n    return x * x",
                "test_list": ["assert square(3) == 9"],
                "test_setup_code": "",
                "challenge_test_list": [],
            }
        ],
        "test": [
            {
                "task_id": 3,
                "text": "Write a function to cube a number.",
                "code": "def cube(x):\n    return x * x * x",
                "test_list": ["assert cube(2) == 8"],
                "test_setup_code": "",
                "challenge_test_list": [],
            }
        ],
    }

    def fake_load_dataset(name, config, split):
        assert name == "google-research-datasets/mbpp"
        assert config == "full"
        return rows[split]

    monkeypatch.setattr("examples.aime_math.utils.load_dataset", fake_load_dataset)

    task = MBPPTask(
        source="huggingface",
        hf_dataset="google-research-datasets/mbpp",
        hf_config="full",
    )
    splits = task.load_splits()

    train_example = splits.trainset[0]
    assert isinstance(train_example, dspy.Example)
    assert train_example.problem_id == 1
    assert train_example.content == "Write a function to add two integers."
    assert train_example.canonical_solution.startswith("def add_numbers")
    assert train_example.function_name == "add_numbers"
    assert train_example.signature_line == "def add_numbers(a, b):"
    assert train_example.test_list == ["assert add_numbers(1, 2) == 3"]
    assert train_example.challenge_test_list == ["assert add_numbers(-1, 1) == 0"]
    assert "Required function name: add_numbers" in train_example.input
    assert 'Function signature: "def add_numbers(a, b):"' in train_example.input

    assert splits.valset[0].problem_id == 2
    assert splits.testset[0].problem_id == 3


def test_mbpp_task_loads_legacy_local_jsonl(tmp_path: Path):
    dspy = pytest.importorskip("dspy")
    from examples.aime_math.utils import MBPPTask

    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    test_path = tmp_path / "test.jsonl"

    payload = (
        '{"id": 11, "content": "Write a function to double a value.", '
        '"canonical_solution": "def double(x):\\n    return x * 2", '
        '"test_list": ["assert double(4) == 8"], '
        '"labels": {"challenge_test_list": ["assert double(-3) == -6"], "test_setup_code": ""}}'
    )
    for path in (train_path, val_path, test_path):
        path.write_text(payload + "\n", encoding="utf-8")

    task = MBPPTask(tmp_path, source="local")
    splits = task.load_splits()

    example = splits.trainset[0]
    assert isinstance(example, dspy.Example)
    assert example.problem_id == 11
    assert example.function_name == "double"
    assert example.signature_line == "def double(x):"
    assert example.challenge_test_list == ["assert double(-3) == -6"]
    assert splits.valset[0].problem_id == 11
    assert splits.testset[0].problem_id == 11
