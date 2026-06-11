#!/usr/bin/env python
"""Evaluate the GEPA *baseline* (initial prompt, no optimization) on the AIME 2025
test set, using OUR pipeline with the repo's DEFAULT parameters.

This mirrors what `report_final_results` does for the baseline, but standalone so you
can get a clean baseline number without running a full optimization.

Defaults (the repo's standard config — NOT changed):
  - solver model  : openai/deepseek-chat        (== DeepSeek V4 Flash)
  - temperature   : 0.2     (DEFAULT_SOLVER_TEMPERATURE)
  - max_tokens    : 32000   (DEFAULT_MAX_TOKENS)
  - dataset split : random.Random(seed), seed = 0
  - test set      : AIME 2025 (30 problems), exact integer match
  - solver path   : dspy.ChainOfThought (AIME_SOLVER_USE_DSPY=1, default);
                    set AIME_SOLVER_USE_DSPY=0 to use the legacy litellm+JSON path.
  - solver cache  : disabled by default (fresh samples); enable with
                    AIME_EVAL_SOLVER_CACHE_ENABLED=1.

Configurable via env (all optional):
  AIME_SEED, AIME_DEEPSEEK_MODEL, DEEPSEEK_API_BASE, AIME_SOLVER_TEMPERATURE,
  AIME_SOLVER_MAX_TOKENS, AIME_MAX_WORKERS, AIME_EVAL_PASS_K,
  AIME_EVAL_SOLVER_CACHE_ENABLED, AIME_SOLVER_USE_DSPY,
  AIME_INITIAL_PROMPT (override the prompt text entirely).

Run:
  cd /home/jlzeng/code/gepa
  DEEPSEEK_API_KEY=... .venv/bin/python examples/aime_math/run_baseline_eval.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    os.chdir(repo_root)
    sys.path.insert(0, str(repo_root))

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("Set DEEPSEEK_API_KEY before running.")

    from examples.aime_math.config import (
        DEFAULT_DEEPSEEK_API_BASE,
        DEFAULT_DEEPSEEK_MODEL,
        DEFAULT_INITIAL_PROMPTS,
        DEFAULT_MAX_TOKENS,
        DEFAULT_SOLVER_TEMPERATURE,
    )
    from examples.aime_math.utils import (
        configure_default_solver_client,
        evaluate_on_dataset,
        load_math_dataset,
    )

    seed = int(os.environ.get("AIME_SEED", "0"))
    solver_model = os.environ.get("AIME_DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL)
    api_base = os.environ.get("DEEPSEEK_API_BASE", DEFAULT_DEEPSEEK_API_BASE)
    temperature = float(os.environ.get("AIME_SOLVER_TEMPERATURE", str(DEFAULT_SOLVER_TEMPERATURE)))
    max_tokens = int(os.environ.get("AIME_SOLVER_MAX_TOKENS", str(DEFAULT_MAX_TOKENS)))
    max_workers = int(os.environ.get("AIME_MAX_WORKERS", "16"))
    pass_k = int(os.environ.get("AIME_EVAL_PASS_K", "1"))
    use_cache = _env_bool("AIME_EVAL_SOLVER_CACHE_ENABLED", False)
    use_dspy = _env_bool("AIME_SOLVER_USE_DSPY", True)

    prompt = os.environ.get("AIME_INITIAL_PROMPT", DEFAULT_INITIAL_PROMPTS["aime"])

    completion_kwargs = {
        "api_key": api_key,
        "api_base": api_base,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }
    cache_namespace = {
        "solver_backend": "litellm_deepseek_json_v1",
        "model": solver_model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "backend": "baseline_eval",
    }
    configure_default_solver_client(
        cache_dir=repo_root / "outputs" / "aime_math" / "baseline_eval" / f"api_cache_seed_{seed}",
        cache_namespace=cache_namespace,
        model_name=solver_model,
        completion_kwargs=completion_kwargs,
        enable_cache=use_cache,
        output_mode="integer",
    )

    trainset, valset, testset = load_math_dataset(seed=seed)

    print("=" * 70)
    print("[BASELINE-EVAL] GEPA baseline (initial prompt, no optimization) on AIME 2025")
    print("=" * 70)
    print(f"  solver model    : {solver_model}")
    print(f"  api_base        : {api_base}")
    print(f"  temperature     : {temperature}")
    print(f"  max_tokens      : {max_tokens}")
    print(f"  solver path     : {'dspy.ChainOfThought' if use_dspy else 'legacy litellm+JSON'}")
    print(f"  pass_k          : {pass_k} (exact-match)")
    print(f"  seed (split)    : {seed}")
    print(f"  solver cache    : {'ENABLED' if use_cache else 'disabled'}")
    print(f"  train/val/test  : {len(trainset)}/{len(valset)}/{len(testset)}")
    print(f"  initial prompt  : {prompt!r}")
    print("-" * 70)

    t0 = time.perf_counter()
    stats = evaluate_on_dataset(
        prompt,
        testset,
        max_workers=max_workers,
        use_solver_cache=use_cache,
        pass_k=pass_k,
        return_stats=True,
        cache_label="baseline_eval",
    )
    elapsed = time.perf_counter() - t0

    print("-" * 70)
    print(f"[BASELINE-EVAL] DONE in {elapsed:.1f}s over {stats['total_examples']} test problems")
    print(f"  pass@{stats['pass_k']}  (exact-match): {stats['pass_score']:.2%}")
    print(f"  mean@{stats['pass_k']} (exact-match): {stats['mean_score']:.2%}")
    print(f"  total solver attempts: {stats.get('total_attempts', 'n/a')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
