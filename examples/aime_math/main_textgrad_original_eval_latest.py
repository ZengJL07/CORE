from __future__ import annotations

from pathlib import Path

from examples.aime_math.config import AIMEExperimentConfig
from examples.aime_math.experiment import AIMEExperiment


REPO_ROOT = Path("/home/jlzeng/code/gepa")
HARDCODED_PROMPT_PATH = REPO_ROOT / "examples" / "aime_math" / "textgrad_original" / "latest_prompt_tgd_21055.txt"
SOURCE_RUN_DIR = REPO_ROOT / "runs" / "textgrad_original"
SOURCE_SEED = 42


def _load_hardcoded_prompt(prompt_path: Path) -> str:
    if not prompt_path.exists():
        raise FileNotFoundError(f"Could not find hardcoded latest prompt at {prompt_path}")
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"Hardcoded latest prompt file is empty: {prompt_path}")
    return prompt


def main() -> None:
    latest_prompt = _load_hardcoded_prompt(HARDCODED_PROMPT_PATH)

    print(f"[AIME] Loaded hardcoded latest TextGrad-original prompt from: {HARDCODED_PROMPT_PATH}")
    print(f"[AIME] Source run directory for artifacts: {SOURCE_RUN_DIR}")
    print(f"[AIME] Source training seed for this prompt: {SOURCE_SEED}")

    config = AIMEExperimentConfig.from_env("textgrad_original")
    experiment = AIMEExperiment(config)
    experiment.print_startup_banner()

    if config.seed != SOURCE_SEED:
        print(
            "[AIME] Warning: current AIME_SEED does not match the hardcoded prompt seed "
            f"({config.seed} != {SOURCE_SEED}). Override AIME_SEED to match if you want strict comparability."
        )

    experiment.report_final_results(
        latest_prompt,
        label="original_tgd_latest",
        output_stem="textgrad_original_tgd_latest_result_plot",
        run_dir=SOURCE_RUN_DIR,
    )


if __name__ == "__main__":
    main()
