from pathlib import Path

from examples.aime_math.config import AIMEExperimentConfig
from examples.aime_math.experiment import AIMEExperiment
from examples.aime_math.run_artifacts import capture_run_logs
from examples.aime_math.sample_weighted_gepa import SampleWeightedGEPAConfig, SampleWeightedGEPARunner


def _load_existing_prompt(run_dir: Path) -> str:
    summary_path = run_dir / "sample_weighted_gepa_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Could not find saved SampleWeightedGEPA summary at {summary_path}")
    import json

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    prompt = payload.get("best_prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"Summary file {summary_path} does not contain a valid best_prompt")
    return prompt


def main() -> None:
    config = AIMEExperimentConfig.from_env("sample_weighted_gepa")
    with capture_run_logs(config.run_dir):
        experiment = AIMEExperiment(config)
        experiment.print_startup_banner()

        if config.evaluate_existing_run_dir is not None:
            optimized_prompt = _load_existing_prompt(config.evaluate_existing_run_dir)
            print("[AIME] Evaluating existing SampleWeightedGEPA prompt from saved run.")
            experiment.report_final_results(
                optimized_prompt,
                label="sample_weighted_gepa_existing",
                output_stem="sample_weighted_gepa_existing_result_plot",
                run_dir=config.evaluate_existing_run_dir,
            )
            return

        algorithm_config = SampleWeightedGEPAConfig.from_env()
        runner = SampleWeightedGEPARunner(experiment, algorithm_config)
        result = runner.run()
        print(
            "[AIME] SampleWeightedGEPA finished "
            f"(best_val_score={result.best_val_score:.2%}, "
            f"metric_calls={result.total_metric_calls}, outer_steps={result.total_outer_steps})."
        )

        experiment.report_final_results(
            result.best_prompt,
            label="sample_weighted_gepa",
            output_stem="sample_weighted_gepa_result_plot",
        )


if __name__ == "__main__":
    main()
