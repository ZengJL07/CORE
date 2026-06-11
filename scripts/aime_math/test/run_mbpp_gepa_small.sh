#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/jlzeng/code/gepa"
TEST_ROOT="$REPO_ROOT/examples/aime_math/test/mbpp_small/gepa"
COMMON_CACHE_ROOT="/home/jlzeng/code/cache/gepa/real/mbpp/shared"
cd "$REPO_ROOT"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

mkdir -p "$TEST_ROOT"

export AIME_DATASET="mbpp"
export AIME_MBPP_SOURCE="${AIME_MBPP_SOURCE:-huggingface}"
export AIME_MBPP_HF_DATASET="${AIME_MBPP_HF_DATASET:-google-research-datasets/mbpp}"
export AIME_MBPP_HF_CONFIG="${AIME_MBPP_HF_CONFIG:-full}"
export AIME_MBPP_DATA_DIR="${AIME_MBPP_DATA_DIR:-/home/jlzeng/code/mbpptest/data}"
export AIME_SEED="${AIME_SEED:-42}"
export AIME_MAX_METRIC_CALLS="${AIME_MAX_METRIC_CALLS:-70}"
export AIME_MAX_WORKERS="${AIME_MAX_WORKERS:-1}"
export AIME_PARALLEL_EVALUATION="${AIME_PARALLEL_EVALUATION:-false}"
export AIME_NUM_PARALLEL_PROPOSALS="${AIME_NUM_PARALLEL_PROPOSALS:-1}"
export AIME_EVAL_PASS_K="${AIME_EVAL_PASS_K:-1}"
export AIME_SKIP_BASELINE_EVAL="${AIME_SKIP_BASELINE_EVAL:-false}"
export AIME_MAX_TRAIN_EXAMPLES="${AIME_MAX_TRAIN_EXAMPLES:-4}"
export AIME_MAX_VAL_EXAMPLES="${AIME_MAX_VAL_EXAMPLES:-4}"
export AIME_MAX_TEST_EXAMPLES="${AIME_MAX_TEST_EXAMPLES:-4}"

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
export AIME_SOLVER_CACHE_DIR="${AIME_SOLVER_CACHE_DIR:-$COMMON_CACHE_ROOT/solver_cache1/seed_${AIME_SEED}}"
export AIME_REFLECTION_CACHE_DIR="${AIME_REFLECTION_CACHE_DIR:-$COMMON_CACHE_ROOT/reflection_cache/seed_${AIME_SEED}}"

python -m examples.aime_math.main
