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


DEFAULT_TRACE2SKILL_DIR = REPO_ROOT / "runs" / "trace2skill_baseline" / "20260504_165508_038531"
DEFAULT_TEXTGRAD_ORIGINAL_DIR = REPO_ROOT / "runs" / "textgrad_original"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "aime_math" / "comparison_plots"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
                "iteration": float(item["iteration"]),
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
                "score": float(best_score),
            }
        )
    return running_best


def _build_method_record(
    *,
    method_key: str,
    label: str,
    summary_path: Path,
    budget_points: list[dict[str, float]],
    budget_note: str,
) -> dict[str, Any]:
    summary = _load_json(summary_path)
    return {
        "method_key": method_key,
        "label": label,
        "summary_path": str(summary_path),
        "budget_points": budget_points,
        "best_so_far_points": _running_best_budget_points(budget_points),
        "budget_note": budget_note,
        "final_metrics": _extract_final_metrics(summary),
    }


def build_payload(trace2skill_dir: Path, textgrad_original_dir: Path) -> dict[str, Any]:
    trace2skill_summary_path = trace2skill_dir / "trace2skill_baseline_result_plot_summary.json"
    textgrad_original_summary_path = textgrad_original_dir / "textgrad_original_tgd_latest_result_plot_summary.json"

    methods = [
        _build_method_record(
            method_key="trace2skill_baseline",
            label="Trace2Skill",
            summary_path=trace2skill_summary_path,
            budget_points=_trace2skill_budget_points(trace2skill_dir),
            budget_note="Budget uses Trace2Skill reported total metric calls from summary.json.",
        ),
        _build_method_record(
            method_key="textgrad_original",
            label="TextGrad",
            summary_path=textgrad_original_summary_path,
            budget_points=_textgrad_original_budget_points(textgrad_original_dir),
            budget_note="Budget uses metric_calls parsed from the textgrad run_log.",
        ),
    ]

    return {
        "trace2skill_dir": str(trace2skill_dir),
        "textgrad_original_dir": str(textgrad_original_dir),
        "methods": methods,
    }


def _write_optimizer_curves(payload: dict[str, Any], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "trace2skill_baseline": "#355070",
        "textgrad_original": "#e76f51",
    }

    fig, (ax_raw, ax_best) = plt.subplots(
        2,
        1,
        figsize=(14.0, 7.0),
        sharex=True,
        constrained_layout=True,
    )

    for method in payload["methods"]:
        xs = [point["budget"] for point in method["budget_points"]]
        ys = [point["score"] for point in method["budget_points"]]
        best_xs = [point["budget"] for point in method["best_so_far_points"]]
        best_ys = [point["score"] for point in method["best_so_far_points"]]

        ax_raw.plot(
            xs,
            ys,
            marker="o",
            linewidth=2.2,
            markersize=5.0,
            alpha=0.85,
            color=colors[method["method_key"]],
            label=method["label"],
        )
        ax_best.step(
            best_xs,
            best_ys,
            where="post",
            linewidth=2.8,
            color=colors[method["method_key"]],
            label=method["label"],
        )

    ax_raw.set_title("Validation Score vs Budget")
    ax_raw.set_ylabel("validation score")
    ax_raw.set_ylim(0.5, 0.82)
    ax_raw.grid(True, alpha=0.25)
    ax_raw.legend(loc="lower right")

    ax_best.set_title("Best Validation Score So Far vs Budget")
    ax_best.set_xlabel("budget (metric calls)")
    ax_best.set_ylabel("validation score")
    ax_best.set_ylim(0.5, 0.82)
    ax_best.grid(True, alpha=0.25)
    ax_best.legend(loc="lower right")

    fig.suptitle("TextGrad vs Trace2Skill Optimization Curves", fontsize=16, fontweight="bold")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot only the TextGrad and Trace2Skill optimization curves used in the AIME comparison figure."
    )
    parser.add_argument("--trace2skill-dir", type=Path, default=DEFAULT_TRACE2SKILL_DIR)
    parser.add_argument("--textgrad-original-dir", type=Path, default=DEFAULT_TEXTGRAD_ORIGINAL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-stem", default="aime_textgrad_trace2skill_curves")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_payload(
        trace2skill_dir=args.trace2skill_dir,
        textgrad_original_dir=args.textgrad_original_dir,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{args.output_stem}.json"
    png_path = args.output_dir / f"{args.output_stem}.png"

    payload["plot_paths"] = {"optimizer_curves": str(png_path)}
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_optimizer_curves(payload, png_path)

    print(f"Wrote comparison data to: {json_path}")
    print(f"Wrote optimizer curves to: {png_path}")


if __name__ == "__main__":
    main()
