import argparse
import concurrent.futures
import json
import os
import re
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


OUTPUT_ROOT = Path("outputs/aime_math")
DEFAULT_PROMPT_PAIRS = Path(__file__).resolve().parent / "prompt_structure_ablation_pairs.json"


@dataclass(frozen=True)
class PromptPair:
    prompt_id: str
    source: str
    structured: str
    flattened: str
    structure_score: int


def _candidate_text(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        candidate = value.get("current_candidate")
        if isinstance(candidate, str):
            return candidate
    return None


def _extract_prompt_from_fitness_cache(payload: dict[str, Any]) -> str | None:
    result = payload.get("data", {}).get("result", [])
    if not isinstance(result, list) or len(result) < 3 or not isinstance(result[2], dict):
        return None
    prompt = result[2].get("prompt")
    return prompt if isinstance(prompt, str) else None


def _list_marker_count(prompt: str) -> int:
    count = 0
    for line in prompt.splitlines():
        stripped = line.strip()
        if re.match(r"^[-*+]\s+\S", stripped):
            count += 1
        elif re.match(r"^\d+[.)]\s+\S", stripped):
            count += 1
    return count


def _load_candidate_prompts(cache_root: Path) -> list[tuple[str, str]]:
    prompts: list[tuple[str, str]] = []

    for path in sorted(cache_root.glob("**/candidates.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, list):
            continue
        for value in data:
            text = _candidate_text(value)
            if text:
                prompts.append((str(path), text))

    for path in sorted((cache_root / "fitness_cache_json").glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        text = _extract_prompt_from_fitness_cache(data)
        if text:
            prompts.append((str(path), text))

    return prompts


def _flatten_prompt(prompt: str) -> str:
    """Remove visible bullet/list structure while preserving the prompt's content."""
    lines: list[str] = []
    for raw_line in prompt.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line in {'"""', "'''", "```"}:
            continue
        line = re.sub(r"^```[a-zA-Z0-9_-]*$", "", line).strip()
        line = re.sub(r"^[-*+]\s+", "", line)
        line = re.sub(r"^\d+[.)]\s+", "", line)
        line = re.sub(r"\*\*(.*?)\*\*", r"\1", line)
        line = re.sub(r"`([^`]*)`", r"\1", line)
        line = line.rstrip(":")
        if line:
            lines.append(line)

    flattened = " ".join(lines)
    flattened = re.sub(r"\s+", " ", flattened).strip()
    return flattened


def build_prompt_pairs(cache_root: Path, limit: int) -> list[PromptPair]:
    seen: set[str] = set()
    candidates: list[PromptPair] = []

    for source, prompt in _load_candidate_prompts(cache_root):
        prompt = prompt.strip()
        if not prompt or prompt in seen:
            continue
        seen.add(prompt)

        score = _list_marker_count(prompt)
        if score < 2:
            continue

        flattened = _flatten_prompt(prompt)
        if not flattened or flattened == prompt:
            continue

        candidates.append(
            PromptPair(
                prompt_id=f"prompt_{len(candidates)}",
                source=source,
                structured=prompt,
                flattened=flattened,
                structure_score=score,
            )
        )

    candidates.sort(key=lambda pair: (pair.structure_score, len(pair.structured)), reverse=True)
    return candidates[:limit]


def load_prompt_pairs(path: Path, limit: int | None = None) -> list[PromptPair]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Prompt pair dataset must be a JSON list: {path}")

    pairs: list[PromptPair] = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Prompt pair item {idx} must be an object.")
        structured = item.get("structured_prompt")
        flattened = item.get("flattened_prompt")
        if not isinstance(structured, str) or not isinstance(flattened, str):
            raise ValueError(f"Prompt pair item {idx} must include structured_prompt and flattened_prompt strings.")
        pairs.append(
            PromptPair(
                prompt_id=str(item.get("prompt_id", f"prompt_{idx}")),
                source=str(item.get("source", path)),
                structured=structured,
                flattened=flattened,
                structure_score=int(item.get("structure_score", _list_marker_count(structured))),
            )
        )

    return pairs[:limit] if limit is not None else pairs


def _score_one(example, prompt: str) -> dict[str, Any]:
    from examples.aime_math.utils import math_metric, run_llm

    prediction = run_llm(example, prompt)
    score, feedback = math_metric(example, prediction)
    return {
        "score": score,
        "answer": getattr(prediction, "answer", ""),
        "reasoning": getattr(prediction, "reasoning", ""),
        "feedback": feedback,
    }


def _evaluate_prompt(prompt: str, examples: list[Any], workers: int) -> list[dict[str, Any]]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_score_one, example, prompt) for example in examples]
        return [future.result() for future in futures]


def _paired_summary(structured_scores: list[float], flattened_scores: list[float]) -> dict[str, Any]:
    diffs = [structured - flattened for structured, flattened in zip(structured_scores, flattened_scores)]
    return {
        "structured_score": sum(structured_scores) / len(structured_scores),
        "flattened_score": sum(flattened_scores) / len(flattened_scores),
        "delta": sum(diffs) / len(diffs),
        "structured_wins": sum(1 for diff in diffs if diff > 0),
        "flattened_wins": sum(1 for diff in diffs if diff < 0),
        "ties": sum(1 for diff in diffs if diff == 0),
    }


def _select_dataset(split: str, seed: int, max_examples: int | None) -> list[Any]:
    from examples.aime_math.utils import load_math_dataset

    trainset, valset, testset = load_math_dataset(seed=seed)
    datasets = {"train": trainset, "val": valset, "test": testset}
    examples = list(datasets[split])
    if max_examples is not None:
        examples = examples[:max_examples]
    return examples


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    import dspy

    from examples.aime_math.utils import migrate_existing_api_caches

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("Please set DEEPSEEK_API_KEY.")

    api_base = os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
    solver_lm = dspy.LM(
        args.model,
        api_key=api_key,
        api_base=api_base,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    dspy.configure(lm=solver_lm)

    migrated, skipped = migrate_existing_api_caches()
    examples = _select_dataset(args.split, args.seed, args.max_examples)
    pairs = load_prompt_pairs(args.prompt_pairs, args.num_prompts) if args.prompt_pairs else build_prompt_pairs(args.cache_root, args.num_prompts)
    if not pairs:
        raise ValueError(f"No structured prompts with at least two bullet/numbered lines found under {args.cache_root}.")

    results: list[dict[str, Any]] = []
    for pair in pairs:
        print(f"[AIME-STRUCTURE] Evaluating {pair.prompt_id} from {pair.source}")
        structured_runs = []
        flattened_runs = []
        for repeat_idx in range(args.repeats):
            print(f"[AIME-STRUCTURE] Repeat {repeat_idx + 1}/{args.repeats}: structured")
            structured = _evaluate_prompt(pair.structured, examples, args.workers)
            print(f"[AIME-STRUCTURE] Repeat {repeat_idx + 1}/{args.repeats}: flattened")
            flattened = _evaluate_prompt(pair.flattened, examples, args.workers)
            structured_runs.append(structured)
            flattened_runs.append(flattened)

        run_summaries = []
        for structured, flattened in zip(structured_runs, flattened_runs):
            run_summaries.append(
                _paired_summary(
                    [item["score"] for item in structured],
                    [item["score"] for item in flattened],
                )
            )

        results.append(
            {
                "prompt_id": pair.prompt_id,
                "source": pair.source,
                "structure_score": pair.structure_score,
                "structured_prompt": pair.structured,
                "flattened_prompt": pair.flattened,
                "run_summaries": run_summaries,
                "mean_delta": statistics.mean(summary["delta"] for summary in run_summaries),
                "raw_runs": {
                    "structured": structured_runs,
                    "flattened": flattened_runs,
                },
            }
        )

    all_deltas = [result["mean_delta"] for result in results]
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "cache_root": str(args.cache_root),
            "prompt_pairs": str(args.prompt_pairs) if args.prompt_pairs else None,
            "split": args.split,
            "seed": args.seed,
            "max_examples": args.max_examples,
            "num_prompts": args.num_prompts,
            "repeats": args.repeats,
            "workers": args.workers,
            "model": args.model,
            "api_base": api_base,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "migrated_cache_entries": migrated,
            "skipped_cache_entries": skipped,
        },
        "aggregate": {
            "mean_delta": statistics.mean(all_deltas),
            "median_delta": statistics.median(all_deltas),
            "structured_better_prompts": sum(1 for delta in all_deltas if delta > 0),
            "flattened_better_prompts": sum(1 for delta in all_deltas if delta < 0),
            "tied_prompts": sum(1 for delta in all_deltas if delta == 0),
        },
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a paired ablation comparing cached structured prompts against flattened prose versions."
    )
    parser.add_argument("--cache-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--prompt-pairs", type=Path, default=DEFAULT_PROMPT_PAIRS)
    parser.add_argument("--num-prompts", type=int, default=5)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--model", default="openai/deepseek-chat")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=32000)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_ROOT / "structure_ablation_results.json",
        help="Path for the JSON results.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print selected prompt pairs without calling the solver API.",
    )
    args = parser.parse_args()

    pairs = load_prompt_pairs(args.prompt_pairs, args.num_prompts) if args.prompt_pairs else build_prompt_pairs(args.cache_root, args.num_prompts)
    source = args.prompt_pairs if args.prompt_pairs else args.cache_root
    print(f"[AIME-STRUCTURE] Selected {len(pairs)} structured prompts from {source}")
    for pair in pairs:
        print(
            f"\n[{pair.prompt_id}] structure_score={pair.structure_score} source={pair.source}\n"
            f"STRUCTURED:\n{pair.structured}\n\nFLATTENED:\n{pair.flattened}\n"
        )

    if args.dry_run:
        return

    result = run_experiment(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[AIME-STRUCTURE] Wrote results to {args.output}")
    print(f"[AIME-STRUCTURE] Aggregate: {json.dumps(result['aggregate'], ensure_ascii=False)}")


if __name__ == "__main__":
    main()
