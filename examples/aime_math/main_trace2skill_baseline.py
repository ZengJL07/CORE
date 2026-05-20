from pathlib import Path

from examples.aime_math.config import AIMEExperimentConfig
from examples.aime_math.experiment import AIMEExperiment
from examples.aime_math.trace2skill_baseline import Trace2SkillBaselineConfig, Trace2SkillBaselineRunner


def _load_existing_prompt(run_dir: Path) -> str:
    prompt_path = run_dir / "trace2skill_baseline" / "optimized_prompt.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Could not find saved Trace2Skill baseline prompt at {prompt_path}")
    return prompt_path.read_text(encoding="utf-8")


def main() -> None:
    config = AIMEExperimentConfig.from_env("trace2skill_baseline")
    experiment = AIMEExperiment(config)
    experiment.print_startup_banner()

    if config.evaluate_existing_run_dir is not None:
        optimized_prompt = _load_existing_prompt(config.evaluate_existing_run_dir)
        print("[AIME] Evaluating existing Trace2Skill baseline prompt from saved run.")
        experiment.report_final_results(
            optimized_prompt,
            label="trace2skill_baseline_existing",
            output_stem="trace2skill_baseline_existing_result_plot",
            run_dir=config.evaluate_existing_run_dir,
        )
        return

    baseline_config = Trace2SkillBaselineConfig.from_env()
    runner = Trace2SkillBaselineRunner(experiment, baseline_config)
    result = runner.run()
    print(
        "[AIME] Trace2Skill baseline finished "
        f"(final_val_score={result.val_score:.2%}, metric_calls_used={result.metric_calls_used}, "
        f"num_iterations={result.num_iterations}, num_suggestions={result.num_suggestions})."
    )

    experiment.report_final_results(
        result.optimized_prompt,
        label="trace2skill_baseline",
        output_stem="trace2skill_baseline_result_plot",
    )


if __name__ == "__main__":
    main()
