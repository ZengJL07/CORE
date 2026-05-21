import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.aime_math.trace2skill_baseline.config import Trace2SkillBaselineConfig
from examples.aime_math.trace2skill_baseline.pipeline import Trace2SkillBaselineRunner


class FakeCollector:
    def __init__(self, experiment, output_dir):
        self.experiment = experiment
        self.output_dir = output_dir

    def collect(self, prompt: str, dataset, budget: int, iteration_idx: int):
        return [
            SimpleNamespace(
                trajectory_id=f"iter_{iteration_idx}_traj_{idx}",
                score=1.0,
            )
            for idx, _example in enumerate(dataset[:budget])
        ]


class FakeAnalystRunner:
    def __init__(self, experiment, output_dir, error_analyst_max_turns: int):
        self.experiment = experiment
        self.output_dir = output_dir
        self.error_analyst_max_turns = error_analyst_max_turns

    def analyze(self, current_prompt: str, trajectories, *, mode: str, iteration_idx: int):
        return [f"suggestion-{iteration_idx}-{idx}" for idx, _ in enumerate(trajectories)]


class FakeMerger:
    def __init__(self, reflection_lm, output_dir, *, task):
        self.reflection_lm = reflection_lm
        self.output_dir = output_dir
        self.task = task

    def merge_into_prompt(self, current_prompt: str, suggestions, *, merge_fanin: int, iteration_idx: int):
        return f"{current_prompt} -> iter {iteration_idx}", [{"round": 1, "count": len(suggestions)}]


class FakeExperiment:
    def __init__(self, run_dir: Path):
        self.config = SimpleNamespace(
            initial_prompt="initial prompt",
            max_metric_calls=10,
            run_dir=run_dir,
            seed=0,
            dataset_name="aime",
        )
        self.task = SimpleNamespace(
            trace2skill_variable_role_description=lambda: "system prompt to a language model",
            trace2skill_trajectory_objective=lambda: "improve task performance",
            trace2skill_success_filter_instruction=lambda: (
                "This trajectory is SUCCESSFUL. Only return a suggestion if there is a concise, "
                "generalizable lesson that is likely to improve future trajectories. Otherwise return NONE."
            ),
            trace2skill_diagnosis_focus_instruction=lambda: (
                "We are interested in diagnosing why the current variable underperformed on this trajectory."
            ),
            trace2skill_suggestion_instruction=lambda: (
                "Turn the diagnosis into one concise, direct improvement suggestion for the prompt. "
                "The suggestion must be generalizable beyond this example. If no safe generalizable lesson exists, return NONE."
            ),
            trace2skill_merge_instruction=lambda: (
                "Below is a small group of trajectory-local improvement suggestions for the prompt.\n"
                "Merge them into one concise, conflict-free, generalizable consolidated suggestion.\n"
                "Prefer recurring patterns. Drop advice that looks too instance-specific or redundant."
            ),
            trace2skill_rewrite_role_description=lambda: "system prompt",
        )
        self.trainset = [SimpleNamespace(input="train-1"), SimpleNamespace(input="train-2")]
        self.valset = [SimpleNamespace(input="val-1"), SimpleNamespace(input="val-2"), SimpleNamespace(input="val-3")]
        self.reflection_lm = object()
        self.eval_calls: list[str] = []

    def evaluate_prompt_summary(self, prompt: str, dataset, *, pass_k: int = 1, cache_label: str | None = None):
        self.eval_calls.append(prompt)
        return {
            "pass_score": 0.5 if len(self.eval_calls) == 1 else 0.6,
            "mean_score": 0.5 if len(self.eval_calls) == 1 else 0.6,
            "total_attempts": len(dataset),
        }


def test_trace2skill_budget_counts_rollout_and_validation(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "examples.aime_math.trace2skill_baseline.pipeline.TrajectoryCollector",
        FakeCollector,
    )
    monkeypatch.setattr(
        "examples.aime_math.trace2skill_baseline.pipeline.ParallelAnalystRunner",
        FakeAnalystRunner,
    )
    monkeypatch.setattr(
        "examples.aime_math.trace2skill_baseline.pipeline.SuggestionMerger",
        FakeMerger,
    )

    experiment = FakeExperiment(tmp_path)
    runner = Trace2SkillBaselineRunner(experiment, Trace2SkillBaselineConfig())

    result = runner.run()

    assert result.metric_calls_used == 10
    assert result.num_trajectories == 4
    assert result.num_iterations == 2
    assert result.optimized_prompt == "initial prompt -> iter 1 -> iter 2"
    assert result.val_score == 0.6
    assert experiment.eval_calls == ["initial prompt -> iter 1", "initial prompt -> iter 1 -> iter 2"]

    summary_path = tmp_path / "trace2skill_baseline" / "summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["budget_cap"] == 10
    assert payload["metric_calls_used"] == 10
    assert payload["optimized_prompt"] == "initial prompt -> iter 1 -> iter 2"
    assert payload["rollout_metric_calls"] == 4
    assert payload["val_metric_calls"] == 6
    assert payload["valset_size"] == 3
    assert payload["final_val_score"] == 0.6
    assert payload["final_val_mean_score"] == 0.6
    assert payload["iterations"] == [
        {
            "iteration": 1,
            "rollout_size": 2,
            "rollout_metric_calls": 2,
            "val_metric_calls": 3,
            "iteration_metric_calls": 5,
            "remaining_budget": 5,
            "num_suggestions": 2,
            "val_score": 0.5,
            "val_mean_score": 0.5,
            "merge_rounds": 1,
        },
        {
            "iteration": 2,
            "rollout_size": 2,
            "rollout_metric_calls": 2,
            "val_metric_calls": 3,
            "iteration_metric_calls": 5,
            "remaining_budget": 0,
            "num_suggestions": 2,
            "val_score": 0.6,
            "val_mean_score": 0.6,
            "merge_rounds": 1,
        },
    ]


def test_trace2skill_prompts_use_task_hooks() -> None:
    from examples.aime_math.trace2skill_baseline.stage2_analysts import ErrorAnalyst, SuccessAnalyst
    from examples.aime_math.trace2skill_baseline.stage3_merge import SuggestionMerger

    task = SimpleNamespace(
        trace2skill_variable_role_description=lambda: "task-specific system prompt",
        trace2skill_trajectory_objective=lambda: "improve held-out task performance",
        trace2skill_success_filter_instruction=lambda: "SUCCESS FILTER",
        trace2skill_diagnosis_focus_instruction=lambda: "DIAGNOSIS FOCUS",
        trace2skill_suggestion_instruction=lambda: "SUGGESTION INSTRUCTION",
        trace2skill_merge_instruction=lambda: "MERGE INSTRUCTION",
        trace2skill_rewrite_role_description=lambda: "task prompt",
    )
    trajectory = SimpleNamespace(
        input_text="Write a function.",
        gold_answer="def foo():\n    pass",
        model_answer="def foo():\n    return None",
        reasoning="",
        score=0.0,
        feedback="Failed unit tests.",
    )

    success_prompt = SuccessAnalyst(object(), task=task)._build_prompt("prompt", trajectory)
    error_prompt = ErrorAnalyst(object(), 2, task=task)._build_diagnosis_prompt("prompt", trajectory)
    suggestion_prompt = ErrorAnalyst(object(), 2, task=task)._build_suggestion_prompt("prompt", "diag", trajectory)
    merge_prompt = SuggestionMerger(object(), Path("/tmp"), task=task)._build_merge_prompt(
        "prompt",
        ["suggestion 1"],
    )
    rewrite_prompt = SuggestionMerger(object(), Path("/tmp"), task=task)._build_rewrite_prompt("prompt", "s")

    assert "task-specific system prompt" in success_prompt
    assert "improve held-out task performance" in success_prompt
    assert "SUCCESS FILTER" in success_prompt
    assert "DIAGNOSIS FOCUS" in error_prompt
    assert "SUGGESTION INSTRUCTION" in suggestion_prompt
    assert "MERGE INSTRUCTION" in merge_prompt
    assert "task prompt" in rewrite_prompt
