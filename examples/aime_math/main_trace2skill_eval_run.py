from __future__ import annotations

import argparse
from pathlib import Path

from examples.aime_math.config import AIMEExperimentConfig
from examples.aime_math.experiment import AIMEExperiment
from examples.aime_math.plot_trace2skill_run import plot_trace2skill_run


def _load_prompt(run_dir: Path) -> str:
    prompt_path = run_dir / "trace2skill_baseline" / "optimized_prompt.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Could not find Trace2Skill prompt at {prompt_path}")
    return prompt_path.read_text(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved Trace2Skill run and draw its validation curve.")
    parser.add_argument("run_dir", type=Path, help="Saved Trace2Skill run directory")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    prompt = _load_prompt(run_dir)

    config = AIMEExperimentConfig.from_env("trace2skill_baseline")
    experiment = AIMEExperiment(config)
    experiment.print_startup_banner()

    print(f"[AIME] Evaluating saved Trace2Skill prompt from: {run_dir}")
    experiment.config = type(config)(
        **{
            **config.__dict__,
            "skip_baseline_eval": True,
        }
    )
    result = experiment.report_final_results(
        prompt,
        label="trace2skill_baseline_existing",
        output_stem="trace2skill_baseline_existing_result_plot",
        run_dir=run_dir,
    )
    curve_path = plot_trace2skill_run(run_dir)

    print(f"[AIME] Validation curve written to: {curve_path}")
    print(
        "[AIME] Evaluation summary: "
        f"optimized_pass@{config.eval_pass_k}={result['optimized']['pass_score']:.2%}"
    )


if __name__ == "__main__":
    main()
