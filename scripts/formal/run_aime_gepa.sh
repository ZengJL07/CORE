#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/jlzeng/code/gepa"
TEST_ROOT="$REPO_ROOT/examples/aime_math/test/aime_formal/gepa"
COMMON_CACHE_ROOT="/home/jlzeng/code/cache/gepa/real/aime/shared"
cd "$REPO_ROOT"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

mkdir -p "$TEST_ROOT"

export AIME_DATASET="aime"
export AIME_SEED="${AIME_SEED:-42}"
export AIME_MAX_METRIC_CALLS="${AIME_MAX_METRIC_CALLS:-500}"
export AIME_MAX_WORKERS="${AIME_MAX_WORKERS:-15}"
export AIME_PARALLEL_EVALUATION="${AIME_PARALLEL_EVALUATION:-true}"
export AIME_NUM_PARALLEL_PROPOSALS="${AIME_NUM_PARALLEL_PROPOSALS:-5}"
export AIME_EVAL_PASS_K="${AIME_EVAL_PASS_K:-3}"
export AIME_SKIP_BASELINE_EVAL="${AIME_SKIP_BASELINE_EVAL:-false}"
export AIME_MAX_TRAIN_EXAMPLES="${AIME_MAX_TRAIN_EXAMPLES:-45}"
export AIME_MAX_VAL_EXAMPLES="${AIME_MAX_VAL_EXAMPLES:-30}"
export AIME_MAX_TEST_EXAMPLES="${AIME_MAX_TEST_EXAMPLES:-30}"
export AIME_ENABLE_HMMT_FEB_2025_TEST="${AIME_ENABLE_HMMT_FEB_2025_TEST:-false}"
export AIME_ENABLE_HMMT_FEB_2026_TEST="${AIME_ENABLE_HMMT_FEB_2026_TEST:-false}"

# Make GEPA's reflection prompt byte-identical to prompt_ucb's (parent history off):
# same feedback rendering + <REFLECTION>/<IMPROVED_PROMPT> output format.
export AIME_GEPA_ALIGN_REFLECTION_WITH_PROMPT_UCB="${AIME_GEPA_ALIGN_REFLECTION_WITH_PROMPT_UCB:-true}"

# Use the legacy litellm + \boxed{} solver path (no JSON envelope), matching math500.
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
export AIME_RUNS_DIR="${AIME_RUNS_DIR:-$TEST_ROOT/runs}"
export AIME_SOLVER_CACHE_DIR="${AIME_SOLVER_CACHE_DIR:-$COMMON_CACHE_ROOT/solver_cache/seed_${AIME_SEED}}"
export AIME_REFLECTION_CACHE_DIR="${AIME_REFLECTION_CACHE_DIR:-$COMMON_CACHE_ROOT/reflection_cache/seed_${AIME_SEED}}"

python -m examples.aime_math.main
