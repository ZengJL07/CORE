from examples.aime_math.config import AIMEExperimentConfig
from examples.aime_math.experiment import AIMEExperiment
from examples.aime_math.textgrad_original import OriginalTextGradConfig, OriginalTextGradRunner


def main() -> None:
    config = AIMEExperimentConfig.from_env("textgrad")
    experiment = AIMEExperiment(config)
    experiment.print_startup_banner()

    algorithm_config = OriginalTextGradConfig.from_env()
    runner = OriginalTextGradRunner(experiment, algorithm_config)
    result = runner.run()
    print(
        f"[AIME] TextGrad optimization finished with final val score "
        f"{'n/a' if result.final_val_score is None else f'{result.final_val_score:.2%}'} "
        f"after {result.total_steps} steps and {result.total_metric_calls} metric calls."
    )

    experiment.report_final_results(
        result.final_prompt,
        label=algorithm_config.algorithm,
        output_stem=f"textgrad_{algorithm_config.algorithm}_result_plot",
    )


if __name__ == "__main__":
    main()
