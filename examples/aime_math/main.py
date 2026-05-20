from examples.aime_math.config import AIMEExperimentConfig
from examples.aime_math.experiment import AIMEExperiment, BestPromptPrinter
from gepa.optimize_anything import optimize_anything


def main() -> None:
    config = AIMEExperimentConfig.from_env("gepa")
    experiment = AIMEExperiment(config)
    experiment.print_startup_banner()

    if config.evaluate_existing_run_dir is not None:
        if config.evaluate_candidate_idx is None:
            raise ValueError("AIME_EVALUATE_CANDIDATE_IDX must be set when AIME_EVALUATE_EXISTING_RUN_DIR is used.")

        optimized_prompt, val_score = experiment.load_existing_candidate_prompt(
            config.evaluate_existing_run_dir,
            config.evaluate_candidate_idx,
        )
        print(
            "[AIME] Evaluating existing candidate from saved run: "
            f"idx={config.evaluate_candidate_idx}, val_score={val_score:.2%}"
        )
        experiment.report_final_results(
            optimized_prompt,
            label=f"gepa_candidate_{config.evaluate_candidate_idx}",
            output_stem=f"candidate_{config.evaluate_candidate_idx}_result_plot",
            run_dir=config.evaluate_existing_run_dir,
        )
        return

    gepa_config = experiment.build_gepa_config(callbacks=[BestPromptPrinter()])
    print("[AIME] Starting GEPA optimization...")
    result = optimize_anything(
        seed_candidate=config.initial_prompt,
        evaluator=experiment.evaluate,
        dataset=experiment.trainset,
        valset=experiment.valset,
        config=gepa_config,
    )
    print("[AIME] GEPA optimization finished.")
    optimized_prompt, selected_candidate_idx = experiment.select_optimized_prompt(result)
    print(f"[AIME] Selected candidate index for final evaluation: {selected_candidate_idx}")

    experiment.report_final_results(
        optimized_prompt,
        label="gepa",
        output_stem="latest_run_result_plot",
    )


if __name__ == "__main__":
    main()
