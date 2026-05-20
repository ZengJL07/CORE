from __future__ import annotations

import json
from pathlib import Path


def plot_trace2skill_run(run_dir: Path) -> Path:
    summary_path = run_dir / "trace2skill_baseline" / "summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    iterations = payload.get("iterations", [])
    if not iterations:
        raise ValueError(f"No iteration history found in {summary_path}")

    xs = [int(item["iteration"]) for item in iterations]
    ys = [float(item["val_score"]) for item in iterations]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    ax.plot(xs, ys, marker="o", linewidth=2.2, color="#355070")
    ax.set_title("Trace2Skill Validation Score by Iteration")
    ax.set_xlabel("iteration")
    ax.set_ylabel("val score")
    ax.set_ylim(max(0.0, min(ys) - 0.05), min(1.0, max(ys) + 0.05))
    ax.grid(True, alpha=0.25)

    for x_value, y_value in zip(xs, ys, strict=True):
        ax.text(x_value, y_value + 0.008, f"{y_value:.2%}", ha="center", va="bottom", fontsize=9)

    output_path = run_dir / "trace2skill_baseline_val_curve.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot Trace2Skill validation-score curve for a run.")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    path = plot_trace2skill_run(args.run_dir)
    print(path)
