from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_aime_config_parses_optional_hmmt_flags(monkeypatch):
    from examples.aime_math.config import AIMEExperimentConfig

    monkeypatch.setenv("AIME_ENABLE_HMMT_FEB_2025_TEST", "true")
    monkeypatch.setenv("AIME_ENABLE_HMMT_FEB_2026_TEST", "false")

    config = AIMEExperimentConfig.from_env("gepa")
    assert config.enable_hmmt_feb_2025_test is True
    assert config.enable_hmmt_feb_2026_test is False


def test_load_extra_math_testset_maps_hmmt_fields(monkeypatch, tmp_path: Path):
    dspy = pytest.importorskip("dspy")
    from examples.aime_math.utils import load_extra_math_testset

    def fake_load_dataset(name, config, split):
        assert name == "MathArena/hmmt_feb_2025"
        assert config == "default"
        assert split == "train"
        return [
            {
                "problem_idx": 101,
                "problem": "Compute 1+1.",
                "answer": 2,
                "problem_type": "algebra",
            }
        ]

    monkeypatch.setattr("examples.aime_math.utils.load_dataset", fake_load_dataset)
    dataset = load_extra_math_testset("MathArena/hmmt_feb_2025", tmp_path / "hmmt.jsonl")

    assert len(dataset) == 1
    example = dataset[0]
    assert isinstance(example, dspy.Example)
    assert example.input == "Compute 1+1."
    assert str(example.answer) == "2"
    assert example.problem_id == 101
    assert example.problem_type == "algebra"
