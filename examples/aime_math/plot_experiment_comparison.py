from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path("/home/jlzeng/code/gepa")
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from gepa.core.state import GEPAState  # noqa: E402


DEFAULT_TRACE2SKILL_DIR = REPO_ROOT / "runs" / "trace2skill_baseline" / "20260504_165508_038531"
DEFAULT_TEXTGRAD_ORIGINAL_DIR = REPO_ROOT / "runs" / "textgrad_original"
DEFAULT_GEPA_DIR = REPO_ROOT / "outputs" / "aime_math" / "gepa" / "runs" / "20260506_081237_761500"
DEFAULT_PROMPT_UCB_DIR = REPO_ROOT / "runs" / "prompt_UCB" / "20260506_030435_154816"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "aime_math" / "comparison_plots"

def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_run_dir(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else REPO_ROOT / path


def _extract_final_metrics(summary: dict[str, Any]) -> dict[str, float]:
    test_results = summary["test_results"]
    return {
        "pass_at_k": float(test_results["optimized_score"]),
        "acc": float(test_results["optimized_mean_score"]),
        "baseline_pass_at_k": float(test_results["baseline_score"]),
        "baseline_acc": float(test_results["baseline_mean_score"]),
        "pass_k": int(test_results["pass_k"]),
    }


def _trace2skill_budget_points(run_dir: Path) -> list[dict[str, float]]:
    payload = _load_json(run_dir / "trace2skill_baseline" / "summary.json")
    points: list[dict[str, float]] = []
    for item in payload.get("iterations", []):
        if "val_score" not in item:
            continue
        budget_used = int(payload["budget_cap"]) - int(item["remaining_budget"])
        points.append(
            {
                "iteration": int(item["iteration"]),
                "budget": float(budget_used),
                "score": float(item["val_score"]),
            }
        )
    return points


def _textgrad_original_budget_points(run_dir: Path) -> list[dict[str, float]]:
    log_path = run_dir / "run_log"
    text = log_path.read_text(encoding="utf-8", errors="replace")

    initial_match = re.search(r"Initial validation score: ([0-9]+(?:\.[0-9]+)?)%", text)
    step_matches = re.finditer(
        r"Step (\d+) finished: val_score=([0-9]+(?:\.[0-9]+)?)%, metric_calls=(\d+)",
        text,
    )

    points: list[dict[str, float]] = []
    if initial_match:
        points.append(
            {
                "iteration": 0.0,
                "budget": 45.0,
                "score": float(initial_match.group(1)) / 100.0,
            }
        )

    for match in step_matches:
        points.append(
            {
                "iteration": float(match.group(1)),
                "budget": float(match.group(3)),
                "score": float(match.group(2)) / 100.0,
            }
        )

    return points


def _gepa_budget_points(run_dir: Path) -> list[dict[str, float]]:
    summary = _load_json(run_dir / "latest_run_result_plot_summary.json")
    state = GEPAState.load(str(run_dir))

    candidate_points = summary.get("candidate_points", [])
    discovery_counts = list(state.num_metric_calls_by_discovery)
    if len(discovery_counts) < len(candidate_points) + 1:
        raise ValueError(
            "GEPA discovery counts are shorter than expected: "
            f"counts={len(discovery_counts)}, candidate_points={len(candidate_points)}"
        )

    points: list[dict[str, float]] = []
    seed_score = float(state.program_full_scores_val_set[0])
    points.append(
        {
            "iteration": 0.0,
            "budget": 0.0,
            "score": seed_score,
        }
    )

    for idx, point in enumerate(candidate_points, start=1):
        full_eval_cost = int(point.get("coverage", point.get("total", 45)))
        budget_after_full_eval = discovery_counts[idx] + full_eval_cost
        points.append(
            {
                "iteration": float(point["iteration"]),
                "budget": float(budget_after_full_eval),
                "score": float(point["score"]),
            }
        )
    return points


def _parent_reflection_budget_points(run_dir: Path) -> list[dict[str, float]]:
    payload = _load_json(run_dir / "parent_reflection_gepa_summary.json")
    history = payload.get("candidate_points", [])
    outer_step_records = payload.get("outer_step_records", [])
    points: list[dict[str, float]] = []

    budget_by_candidate_iteration: dict[int, float] = {}
    running_candidate_iteration = 0
    for record in outer_step_records:
        total_metric_calls = record.get("total_metric_calls")
        inserted_candidate_ids = record.get("inserted_candidate_ids", [])
        if total_metric_calls is None:
            continue
        for _candidate_id in inserted_candidate_ids:
            running_candidate_iteration += 1
            budget_by_candidate_iteration[running_candidate_iteration] = float(total_metric_calls)

    for idx, point in enumerate(history, start=1):
        if "score" not in point:
            continue
        budget_after_full_eval = budget_by_candidate_iteration.get(idx)
        if budget_after_full_eval is None:
            continue
        points.append(
            {
                "iteration": float(point.get("iteration", idx)),
                "budget": float(budget_after_full_eval),
                "score": float(point["score"]),
            }
        )
    return points


def _build_method_record(
    *,
    method_key: str,
    label: str,
    summary_path: Path,
    budget_points: list[dict[str, float]],
    budget_note: str,
) -> dict[str, Any]:
    summary = _load_json(summary_path)
    final_metrics = _extract_final_metrics(summary)
    return {
        "method_key": method_key,
        "label": label,
        "summary_path": str(summary_path),
        "budget_points": budget_points,
        "best_so_far_points": _running_best_budget_points(budget_points),
        "budget_note": budget_note,
        "final_metrics": final_metrics,
    }


def _running_best_budget_points(points: list[dict[str, float]]) -> list[dict[str, float]]:
    running_best: list[dict[str, float]] = []
    best_score: float | None = None
    for point in points:
        score = float(point["score"])
        if best_score is None or score > best_score:
            best_score = score
        running_best.append(
            {
                "iteration": float(point["iteration"]),
                "budget": float(point["budget"]),
                "score": best_score,
            }
        )
    return running_best


def _average_baseline_metrics(methods: list[dict[str, Any]]) -> dict[str, float]:
    if not methods:
        raise ValueError("Cannot average baseline metrics from an empty method list.")

    baseline_pass_values = [float(method["final_metrics"]["baseline_pass_at_k"]) for method in methods]
    baseline_acc_values = [float(method["final_metrics"]["baseline_acc"]) for method in methods]
    pass_k = int(methods[0]["final_metrics"]["pass_k"])

    return {
        "pass_at_k": sum(baseline_pass_values) / len(baseline_pass_values),
        "acc": sum(baseline_acc_values) / len(baseline_acc_values),
        "pass_k": pass_k,
    }



def _prepend_method_initial_points(methods: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized_methods: list[dict[str, Any]] = []

    for method in methods:
        points = list(method["budget_points"])
        summary = _load_json(Path(method["summary_path"]))
        initial_point: dict[str, float] | None = None

        if method["method_key"] == "gepa":
            state = GEPAState.load(str(_resolve_run_dir(summary["run_dir"])))
            initial_point = {
                "iteration": 0.0,
                "budget": 0.0,
                "score": float(state.program_full_scores_val_set[0]),
            }
            gepa = float(state.program_full_scores_val_set[0])

        elif method["method_key"] == "prompt_ucb":
            training_summary = _load_json(_resolve_run_dir(summary["run_dir"]) / "parent_reflection_gepa_summary.json")
            for candidate in training_summary.get("candidate_pool", []):
                if isinstance(candidate, dict) and int(candidate.get("candidate_id", -1)) == 0:
                    initial_point = {
                        "iteration": 0.0,
                        "budget": 0.0,
                        "score": float(candidate["val_score"]),
                    }
                    break


        normalized_points = [dict(initial_point)] + points if initial_point is not None else points
        normalized_method = dict(method)
        normalized_method["budget_points"] = normalized_points
        normalized_method["best_so_far_points"] = _running_best_budget_points(normalized_points)
        normalized_method["initial_point"] = initial_point
        normalized_methods.append(normalized_method)

    return normalized_methods


def build_comparison_payload(
    trace2skill_dir: Path,
    textgrad_original_dir: Path,
    gepa_dir: Path,
    prompt_ucb_dir: Path,
) -> dict[str, Any]:
    trace2skill_summary_path = trace2skill_dir / "trace2skill_baseline_result_plot_summary.json"
    textgrad_original_summary_path = textgrad_original_dir / "textgrad_original_tgd_latest_result_plot_summary.json"
    gepa_summary_path = gepa_dir / "latest_run_result_plot_summary.json"
    prompt_ucb_summary_path = prompt_ucb_dir / "parent_reflection_gepa_result_plot_summary.json"

    methods = [
        _build_method_record(
            method_key="trace2skill_baseline",
            label="Trace2Skill Baseline",
            summary_path=trace2skill_summary_path,
            budget_points=_trace2skill_budget_points(trace2skill_dir),
            budget_note="Budget uses Trace2Skill's reported total metric calls from summary.json (rollout plus per-iteration valset evaluation).",
        ),
        _build_method_record(
            method_key="textgrad_original",
            label="TextGrad Original",
            summary_path=textgrad_original_summary_path,
            budget_points=_textgrad_original_budget_points(textgrad_original_dir),
            budget_note="Budget uses metric_calls parsed from the available top-level run_log; trajectory may be partial if that log is truncated.",
        ),
        _build_method_record(
            method_key="gepa",
            label="GEPA",
            summary_path=gepa_summary_path,
            budget_points=_gepa_budget_points(gepa_dir),
            budget_note="Budget includes the seed prompt's initial full-val evaluation, then uses GEPA discovery_eval_counts plus full valset evaluation cost for each new candidate point.",
        ),
        _build_method_record(
            method_key="prompt_ucb",
            label="CORE",
            summary_path=prompt_ucb_summary_path,
            budget_points=_parent_reflection_budget_points(prompt_ucb_dir),
            budget_note="Budget uses parent_reflection_gepa candidate evaluation count times valset size for each accepted/evaluated prompt candidate.",
        ),
    ]

    methods = _prepend_method_initial_points(methods)

    return {
        "trace2skill_dir": str(trace2skill_dir),
        "textgrad_original_dir": str(textgrad_original_dir),
        "gepa_dir": str(gepa_dir),
        "prompt_ucb_dir": str(prompt_ucb_dir),
        "metric_definition": {
            "acc": "optimized_mean_score from final test evaluation (equivalent to mean@pass_k in current AIME code).",
            "pass_at_k": "optimized_score from final test evaluation.",
        },
        "average_baseline_metrics": _average_baseline_metrics(methods),
        "methods": methods,
    }


def _write_combined_plot(payload: dict[str, Any], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = payload["methods"]
    average_baseline = payload["average_baseline_metrics"]
    colors = {
        "trace2skill_baseline": "#355070",
        "textgrad_original": "#e76f51",
        "gepa": "#2a9d8f",
        "prompt_ucb": "#f4a261",
    }

    fig = plt.figure(figsize=(15.2, 8.4), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.45, 1.05], height_ratios=[1.0, 1.0])
    ax_raw = fig.add_subplot(grid[0, 0])
    ax_best = fig.add_subplot(grid[1, 0], sharex=ax_raw, sharey=ax_raw)
    ax_acc = fig.add_subplot(grid[0, 1])
    ax_pass = fig.add_subplot(grid[1, 1], sharex=ax_acc)

    for method in methods:
        xs = [point["budget"] for point in method["budget_points"]]
        ys = [point["score"] for point in method["budget_points"]]
        best_xs = [point["budget"] for point in method["best_so_far_points"]]
        best_ys = [point["score"] for point in method["best_so_far_points"]]
        if not xs:
            continue
        ax_raw.plot(
            xs,
            ys,
            marker="o",
            linewidth=2.0,
            markersize=5.2,
            alpha=0.78,
            color=colors[method["method_key"]],
            label=method["label"],
        )
        ax_best.step(
            best_xs,
            best_ys,
            where="post",
            linewidth=2.6,
            color=colors[method["method_key"]],
            label=method["label"],
        )

    ax_raw.set_title("Validation Score vs Budget")
    ax_raw.set_xlabel("budget (metric calls)")
    ax_raw.set_ylabel("validation score")
    ax_raw.set_ylim(0.5, 0.82)
    ax_raw.grid(True, alpha=0.25)
    ax_raw.legend(loc="lower right")

    ax_best.set_title("Best Validation Score So Far vs Budget")
    ax_best.set_xlabel("budget (metric calls)")
    ax_best.set_ylabel("validation score")
    ax_best.grid(True, alpha=0.25)
    ax_best.legend(loc="lower right")

    labels = ["Average Baseline"] + [method["label"] for method in methods]
    acc_values = [average_baseline["acc"]] + [method["final_metrics"]["acc"] for method in methods]
    pass_values = [average_baseline["pass_at_k"]] + [method["final_metrics"]["pass_at_k"] for method in methods]
    pass_k = average_baseline["pass_k"]

    x_positions = list(range(len(labels)))
    bar_colors = ["#6c757d"] + [colors[method["method_key"]] for method in methods]

    acc_bars = ax_acc.bar(
        x_positions,
        acc_values,
        width=0.72,
        color=bar_colors,
    )
    pass_bars = ax_pass.bar(
        x_positions,
        pass_values,
        width=0.72,
        color=bar_colors,
    )

    ax_acc.set_title(f"Final Test acc (mean@{pass_k})")
    ax_acc.set_xticks(x_positions)
    ax_acc.set_xticklabels(labels, rotation=15, ha="right")
    ax_acc.set_ylabel("score")
    ax_acc.set_ylim(0.0, max(acc_values) + 0.18)
    ax_acc.grid(True, axis="y", alpha=0.25)

    ax_pass.set_title(f"Final Test pass@{pass_k}")
    ax_pass.set_xticks(x_positions)
    ax_pass.set_xticklabels(labels, rotation=15, ha="right")
    ax_pass.set_ylabel("score")
    ax_pass.set_ylim(0.0, max(pass_values) + 0.18)
    ax_pass.grid(True, axis="y", alpha=0.25)

    for bar in acc_bars:
        value = bar.get_height()
        ax_acc.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.012,
            f"{value:.1%}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    for bar in pass_bars:
        value = bar.get_height()
        ax_pass.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.012,
                f"{value:.1%}",
                ha="center",
                va="bottom",
                fontsize=10,
        )

    fig.suptitle("AIME Experiment Comparison", fontsize=16, fontweight="bold")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot score-budget and final metric comparisons for AIME experiments.")
    parser.add_argument("--trace2skill-dir", type=Path, default=DEFAULT_TRACE2SKILL_DIR)
    parser.add_argument("--textgrad-original-dir", type=Path, default=DEFAULT_TEXTGRAD_ORIGINAL_DIR)
    parser.add_argument("--gepa-dir", type=Path, default=DEFAULT_GEPA_DIR)
    parser.add_argument("--prompt-ucb-dir", type=Path, default=DEFAULT_PROMPT_UCB_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-stem", default="aime_experiment_comparison")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_comparison_payload(
        trace2skill_dir=args.trace2skill_dir,
        textgrad_original_dir=args.textgrad_original_dir,
        gepa_dir=args.gepa_dir,
        prompt_ucb_dir=args.prompt_ucb_dir,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{args.output_stem}.json"
    combined_png_path = args.output_dir / f"{args.output_stem}.png"

    payload["plot_paths"] = {
        "combined_plot": str(combined_png_path),
    }

    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_combined_plot(payload, combined_png_path)

    print(f"Wrote comparison data to: {json_path}")
    print(f"Wrote combined plot to: {combined_png_path}")


if __name__ == "__main__":
    main()
