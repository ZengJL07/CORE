#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/jlzeng/code/gepa"
TEST_ROOT="$REPO_ROOT/examples/aime_math/test/math500_small/prompt_ucb"
COMMON_CACHE_ROOT="/home/jlzeng/code/cache/gepa/real/math500/shared"
cd "$REPO_ROOT"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

mkdir -p "$TEST_ROOT"

export AIME_DATASET="math500"
export AIME_SEED="${AIME_SEED:-42}"
export AIME_MAX_METRIC_CALLS="${AIME_MAX_METRIC_CALLS:-24}"
export AIME_MAX_WORKERS="${AIME_MAX_WORKERS:-1}"
export AIME_PARALLEL_EVALUATION="${AIME_PARALLEL_EVALUATION:-false}"
export AIME_EVAL_PASS_K="${AIME_EVAL_PASS_K:-1}"
export AIME_SKIP_BASELINE_EVAL="${AIME_SKIP_BASELINE_EVAL:-false}"

# MATH-500 ships a single 500-example split; carve a small train/val out of the
# front (seeded shuffle) and bound train/val/test for a quick smoke run.
export AIME_MATH500_TRAIN_SIZE="${AIME_MATH500_TRAIN_SIZE:-4}"
export AIME_MATH500_VAL_SIZE="${AIME_MATH500_VAL_SIZE:-4}"
export AIME_MAX_TRAIN_EXAMPLES="${AIME_MAX_TRAIN_EXAMPLES:-4}"
export AIME_MAX_VAL_EXAMPLES="${AIME_MAX_VAL_EXAMPLES:-4}"
export AIME_MAX_TEST_EXAMPLES="${AIME_MAX_TEST_EXAMPLES:-4}"

export AIME_PRG_CANDIDATE_POOL_SIZE="${AIME_PRG_CANDIDATE_POOL_SIZE:-20}"
export AIME_PRG_REFLECT_TRAIN_BATCH_SIZE="${AIME_PRG_REFLECT_TRAIN_BATCH_SIZE:-3}"
export AIME_PRG_PROBE_VAL_BATCH_SIZE="${AIME_PRG_PROBE_VAL_BATCH_SIZE:-0}"
export AIME_PRG_MAX_OUTER_STEPS="${AIME_PRG_MAX_OUTER_STEPS:-4}"
export AIME_PRG_NUM_PARALLEL_BRANCHES="${AIME_PRG_NUM_PARALLEL_BRANCHES:-1}"

# prompt-UCB sampling strategy: child_prediction (new, default) vs self_score (legacy).
# Set AIME_PRG_PROMPT_UCB_USE_CHILD_PREDICTION=false to fall back to the legacy z-score-on-own-score version.
export AIME_PRG_PROMPT_UCB_USE_CHILD_PREDICTION="${AIME_PRG_PROMPT_UCB_USE_CHILD_PREDICTION:-true}"
export AIME_PRG_PROMPT_SHRINKAGE_PRIOR="${AIME_PRG_PROMPT_SHRINKAGE_PRIOR:-1.0}"
export AIME_PRG_PROMPT_SCORE_Z_CLIP="${AIME_PRG_PROMPT_SCORE_Z_CLIP:-3.0}"

# Disable parent reflection so prompt_ucb mutates from the current prompt + feedback only,
# matching GEPA's reflection inputs (no parent prompt/reflection history injected).
export AIME_PRG_USE_PARENT_HISTORY="${AIME_PRG_USE_PARENT_HISTORY:-false}"

# Use the legacy litellm + \boxed{} solver path (no JSON envelope).
export AIME_SOLVER_USE_DSPY="${AIME_SOLVER_USE_DSPY:-0}"

export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:?Please set DEEPSEEK_API_KEY}"
export DEEPSEEK_API_BASE="${DEEPSEEK_API_BASE:-https://api.deepseek.com/v1}"
export AIME_DEEPSEEK_MODEL="${AIME_DEEPSEEK_MODEL:-openai/deepseek-chat}"
export AIME_REFLECTION_MODEL="${AIME_REFLECTION_MODEL:-deepseek-v4-pro}"

# Number of attempts on transient API failures (connection resets, 5xx, SSL EOF, rate limits).
export AIME_SOLVER_API_MAX_RETRIES="${AIME_SOLVER_API_MAX_RETRIES:-5}"
export AIME_REFLECTION_API_MAX_RETRIES="${AIME_REFLECTION_API_MAX_RETRIES:-5}"

export AIME_SOLVER_CACHE_ENABLED="${AIME_SOLVER_CACHE_ENABLED:-true}"
export AIME_EVAL_SOLVER_CACHE_ENABLED="${AIME_EVAL_SOLVER_CACHE_ENABLED:-true}"
export AIME_OUTPUT_ROOT="${AIME_OUTPUT_ROOT:-$TEST_ROOT}"
export AIME_SOLVER_CACHE_DIR="${AIME_SOLVER_CACHE_DIR:-$COMMON_CACHE_ROOT/solver_cache/seed_${AIME_SEED}}"
export AIME_REFLECTION_CACHE_DIR="${AIME_REFLECTION_CACHE_DIR:-$COMMON_CACHE_ROOT/reflection_cache/seed_${AIME_SEED}}"
export AIME_RUNS_DIR="${AIME_RUNS_DIR:-$TEST_ROOT/runs}"

python -m examples.aime_math.main_parent_reflection_gepa
