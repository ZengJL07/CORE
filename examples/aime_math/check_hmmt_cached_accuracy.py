from __future__ import annotations

import argparse
import json
from pathlib import Path

import dspy

from examples.aime_math.utils import HMMT_FEB_2025_TEST_JSONL, _prediction_from_payload, math_metric


DEFAULT_PROMPT = "Try to solve the math problem carefully. Break down the steps and provide the final answer as a single exact answer."
DEFAULT_CACHE_DIR = Path("/home/jlzeng/code/cache/gepa/real/aime/shared/solver_cache/seed_42")


def _load_hmmt_examples() -> dict[str, dict]:
    rows = []
    with HMMT_FEB_2025_TEST_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return {str(row["input"]): row for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-score cached HMMT Feb 2025 solver outputs using the current math_metric."
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Directory containing cached solver JSON payloads.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Exact raw prompt string whose cached outputs should be evaluated.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit on number of scored HMMT examples (0 means all).",
    )
    args = parser.parse_args()

    hmmt_by_input = _load_hmmt_examples()
    scored_rows: list[dict[str, object]] = []

    for path in sorted(args.cache_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if str(payload.get("prompt", "")) != args.prompt:
            continue

        problem_input = str(payload.get("input", ""))
        row = hmmt_by_input.get(problem_input)
        if row is None:
            continue

        prediction = _prediction_from_payload(payload, output_mode=str(payload.get("output_mode", "integer")))
        example = dspy.Example(input=problem_input, answer=row["answer"]).with_inputs("input")
        score, feedback = math_metric(example, prediction)
        scored_rows.append(
            {
                "problem_id": row.get("problem_id"),
                "score": float(score),
                "gold": row["answer"],
                "pred": str(prediction.answer),
                "cache_file": path.name,
                "feedback": feedback,
            }
        )
        if args.limit and len(scored_rows) >= args.limit:
            break

    total = len(scored_rows)
    correct = sum(1 for row in scored_rows if float(row["score"]) >= 1.0)
    accuracy = (correct / total) if total else 0.0

    print(
        json.dumps(
            {
                "cache_dir": str(args.cache_dir),
                "prompt": args.prompt,
                "num_scored": total,
                "num_correct": correct,
                "accuracy": accuracy,
                "rows": scored_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
