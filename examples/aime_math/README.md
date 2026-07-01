# AIME Math

Optimize a math-solving prompt for AIME competition problems. The solver and reflection LMs both use the DeepSeek API. The example now supports:

- `GEPA` via `optimize_anything`
- `TextGrad`-style prompt optimization
- `TSGD-M` as a momentum-augmented TextGrad variant
- `Trace2Skill`-style paper baseline with strict rollout budget `B=6`

## Dataset

- **Train + Val**: `AI-MO/aimo-validation-aime` (AIME 2022–2024), split 50/50
- **Test**: `MathArena/aime_2025` (AIME 2025)
- **Optional extra tests**: `MathArena/hmmt_feb_2025`, `MathArena/hmmt_feb_2026` (enabled via boolean flags)

### MATH-500

Set `AIME_DATASET=math500` to optimize on the [`HuggingFaceH4/MATH-500`](https://huggingface.co/datasets/HuggingFaceH4/MATH-500)
benchmark instead of AIME. MATH-500 ships a single 500-example `test` split, so we
shuffle it with `AIME_SEED` and carve our own train/val/test:

- `AIME_MATH500_TRAIN_SIZE` (default `200`) and `AIME_MATH500_VAL_SIZE` (default `100`)
  count from the front of the shuffled list; the remainder becomes the test set.
- Answers may be non-integer (fractions, radicals, `\frac{14}{3}`, ...); the math
  metric already matches general exact-math expressions, not just integers.
- The HMMT extra-test flags only apply to `AIME_DATASET=aime`.

## Setup

From the repo root (`gepa/`):

```bash
uv venv
uv pip install datasets dspy litellm
uv pip install -e .  # must come after dspy to avoid PyPI overwrite
```

## Run

```bash
export DEEPSEEK_API_KEY=...
export DEEPSEEK_API_BASE="${DEEPSEEK_API_BASE:-https://api.deepseek.com/v1}"
uv run python -m examples.aime_math.main
```

Or switch to the TextGrad-style runner:

```bash
export DEEPSEEK_API_KEY=...
export DEEPSEEK_API_BASE="${DEEPSEEK_API_BASE:-https://api.deepseek.com/v1}"
export AIME_TEXTGRAD_ALGORITHM=tgd      # or tsgd_m
uv run python -m examples.aime_math.main_textgrad
```

Launcher scripts are kept outside the repository in the local `scripts/` workspace. The module-based commands above are the canonical in-repo entry points.

All runners share the same dataset split, solver setup, and final AIME 2025 evaluation pipeline so the resulting comparisons are directly attributable to the optimizer.

To additionally evaluate on HMMT benchmarks without changing the default AIME 2025 test set, enable either flag:

```bash
export AIME_ENABLE_HMMT_FEB_2025_TEST=true
export AIME_ENABLE_HMMT_FEB_2026_TEST=true
```

Leaving both flags unset or `false` preserves the original behavior.

## Trace2Skill Baseline Notes

- Reuses the same initial prompt as the GEPA runner from [`config.py`](./config.py).
- Samples each training problem exactly once in Stage 1, then fuses all trajectory-local suggestions together.
- Uses `AIME_TRACE2SKILL_B` as the hierarchical merge fan-in `B` and defaults it to `6`.
- Treats trajectory-local outputs as free-form improvement suggestions, then hierarchically merges them into one rewritten prompt.
- Keeps the implementation isolated under [`trace2skill_baseline/`](./trace2skill_baseline) so existing GEPA and TextGrad paths are unchanged.
