from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path("/home/jlzeng/code/gepa")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from examples.aime_math.config import AIMEExperimentConfig
from examples.aime_math.experiment import AIMEExperiment


def main() -> None:
    config = AIMEExperimentConfig.from_env("gepa")
    experiment = AIMEExperiment(config)

    example = experiment.trainset[0]
    score, side_info = experiment.evaluate(experiment.config.initial_prompt, example)

    reflection_prompt = (
        "I am optimizing a parameter.\n"
        "<REFLECTION>placeholder</REFLECTION>\n"
        "<IMPROVED_PROMPT>placeholder</IMPROVED_PROMPT>"
    )
    reflection_response = experiment.reflection_lm(reflection_prompt)

    print(
        json.dumps(
            {
                "dataset": config.dataset_name,
                "solver_model": config.solver_model,
                "reflection_model": config.reflection_model,
                "score": score,
                "side_info": side_info,
                "reflection_response": reflection_response,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
