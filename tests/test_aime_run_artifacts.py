import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.aime_math.run_artifacts import (
    build_best_events,
    capture_run_logs,
    infer_valset_total_from_summary,
    load_sample_weighted_gepa_candidate_points,
)


def test_load_sample_weighted_gepa_candidate_points_reconstructs_legacy_history() -> None:
    payload = {
        "candidate_pool": [
            {"candidate_id": 1, "prompt": "child one", "val_scores_by_sample": {"a": 1.0, "b": 0.0}},
        ],
        "outer_step_records": [
            {
                "outer_step": 1,
                "sampled_current_candidate_ids": [0],
                "inserted_candidate_ids": [1],
                "branch_records": [
                    {
                        "branch_idx": 1,
                        "current_candidate_id": 0,
                        "current_prompt": "initial prompt",
                        "final_prompt": "child one",
                        "passed_total_gate": True,
                        "total_score": 0.61,
                        "full_val_score": 0.62,
                        "accepted_to_pool": True,
                    }
                ],
                "metadata": {"pool_candidate_ids": [1, 0]},
            },
            {
                "outer_step": 2,
                "sampled_current_candidate_ids": [1, 0],
                "inserted_candidate_ids": [2],
                "branch_records": [
                    {
                        "branch_idx": 1,
                        "current_candidate_id": 1,
                        "current_prompt": "child one",
                        "final_prompt": "child one",
                        "passed_total_gate": True,
                        "total_score": 0.70,
                        "full_val_score": 0.62,
                        "accepted_to_pool": True,
                    },
                    {
                        "branch_idx": 2,
                        "current_candidate_id": 0,
                        "current_prompt": "initial prompt",
                        "final_prompt": "child two",
                        "passed_total_gate": True,
                        "total_score": 0.55,
                        "full_val_score": 0.56,
                        "accepted_to_pool": False,
                    },
                ],
                "metadata": {"pool_candidate_ids": [1, 0]},
            },
        ],
    }

    points = load_sample_weighted_gepa_candidate_points(payload)

    assert points == [
        {
            "iteration": 1,
            "outer_step": 1,
            "branch_idx": 1,
            "candidate_id": 1,
            "score": 0.62,
            "accepted_to_pool": True,
        },
        {
            "iteration": 2,
            "outer_step": 2,
            "branch_idx": 2,
            "candidate_id": 2,
            "score": 0.56,
            "accepted_to_pool": False,
        },
    ]

    assert infer_valset_total_from_summary(payload) == 2
    assert build_best_events(
        [
            {"iteration": 1, "score": 0.62},
            {"iteration": 2, "score": 0.56},
            {"iteration": 3, "score": 0.75},
        ]
    ) == [
        {"iteration": 1, "score": 0.62},
        {"iteration": 3, "score": 0.75},
    ]


def test_load_sample_weighted_gepa_candidate_points_prefers_explicit_history() -> None:
    payload = {
        "candidate_points": [
            {
                "iteration": 4,
                "outer_step": 2,
                "branch_idx": 3,
                "candidate_id": 9,
                "score": 0.88,
                "accepted_to_pool": True,
            }
        ]
    }

    assert load_sample_weighted_gepa_candidate_points(payload) == payload["candidate_points"]


def test_capture_run_logs_writes_stdout_and_stderr(tmp_path) -> None:
    run_dir = tmp_path / "artifacts"

    with capture_run_logs(run_dir):
        print("hello stdout")
        print("hello stderr", file=sys.stderr)

    assert "hello stdout" in (run_dir / "run_log.txt").read_text(encoding="utf-8")
    assert "hello stderr" in (run_dir / "run_log_stderr.txt").read_text(encoding="utf-8")
