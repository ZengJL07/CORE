from pathlib import Path

from examples.aime_math.config import AIMEExperimentConfig
from examples.aime_math.experiment import AIMEExperiment
from examples.aime_math.parent_reflection_gepa import ParentReflectionGEPAConfig, ParentReflectionGEPARunner
from examples.aime_math.run_artifacts import capture_run_logs


def _load_existing_prompt(run_dir: Path) -> str:
    summary_path = run_dir / "parent_reflection_gepa_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Could not find saved ParentReflectionGEPA summary at {summary_path}")
    import json

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    prompt = payload.get("best_prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"Summary file {summary_path} does not contain a valid best_prompt")
    return prompt


def main() -> None:
    config = AIMEExperimentConfig.from_env("parent_reflection_gepa")
    with capture_run_logs(config.run_dir):
        experiment = AIMEExperiment(config)
        experiment.print_startup_banner()

        if config.evaluate_existing_run_dir is not None:
            optimized_prompt = _load_existing_prompt(config.evaluate_existing_run_dir)
            print("[AIME] Evaluating existing ParentReflectionGEPA prompt from saved run.")
            experiment.report_final_results(
                optimized_prompt,
                label="parent_reflection_gepa_existing",
                output_stem="parent_reflection_gepa_existing_result_plot",
                run_dir=config.evaluate_existing_run_dir,
            )
            return

        algorithm_config = ParentReflectionGEPAConfig.from_env()
        runner = ParentReflectionGEPARunner(experiment, algorithm_config)
        result = runner.run()
        print(
            "[AIME] ParentReflectionGEPA finished "
            f"(best_val_score={result.best_val_score:.2%}, "
            f"metric_calls={result.total_metric_calls}, outer_steps={result.total_outer_steps})."
        )

        experiment.report_final_results(
            result.best_prompt,
            label="parent_reflection_gepa",
            output_stem="parent_reflection_gepa_result_plot",
        )


if __name__ == "__main__":
    main()
