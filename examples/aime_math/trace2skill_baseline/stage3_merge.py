from __future__ import annotations

from pathlib import Path

from examples.aime_math.trace2skill_baseline.types import AnalystSuggestion
from examples.aime_math.trace2skill_baseline.utils import extract_tagged_block, write_json


class SuggestionMerger:
    def __init__(self, lm, output_dir: Path):
        self.lm = lm
        self.output_dir = output_dir

    def merge_into_prompt(
        self,
        current_prompt: str,
        suggestions: list[AnalystSuggestion],
        merge_fanin: int,
        iteration_idx: int,
    ) -> tuple[str, list[dict[str, object]]]:
        if not suggestions:
            return current_prompt, []

        merge_history: list[dict[str, object]] = []
        active = [item.suggestion for item in suggestions]
        level = 0

        while len(active) > 1:
            next_level = []
            for group_start in range(0, len(active), merge_fanin):
                group = active[group_start : group_start + merge_fanin]
                response = self.lm(
                    f"""You are consolidating prompt-improvement suggestions in a Trace2Skill-style baseline.

Current prompt:
<prompt>
{current_prompt}
</prompt>

Below is a small group of trajectory-local improvement suggestions for the prompt.
Merge them into one concise, conflict-free, generalizable consolidated suggestion.
Prefer recurring patterns. Drop advice that looks too instance-specific or redundant.

Suggestions:
{self._format_group(group)}

Return only:
<merged_suggestion>...</merged_suggestion>
"""
                )
                merged = extract_tagged_block(response, "merged_suggestion") or response.strip()
                if not merged:
                    merged = "\n".join(group)
                next_level.append(merged.strip())
                merge_history.append(
                    {
                        "level": level,
                        "group_start": group_start,
                        "group_size": len(group),
                        "input_suggestions": group,
                        "merged_suggestion": merged.strip(),
                    }
                )
            active = next_level
            level += 1

        final_suggestion = active[0]
        rewrite_response = self.lm(
            f"""You are rewriting a math-solving prompt after Trace2Skill-style consolidation.

Current prompt:
<prompt>
{current_prompt}
</prompt>

Consolidated improvement suggestion:
<suggestion>
{final_suggestion}
</suggestion>

Rewrite the current prompt so it incorporates the consolidated suggestion while staying concise and directly usable as
the next system prompt. Keep the result as a single prompt string, not a skill file or multi-section document.

Return only:
<prompt>...</prompt>
"""
        )
        rewritten_prompt = extract_tagged_block(rewrite_response, "prompt") or rewrite_response.strip()
        if not rewritten_prompt:
            rewritten_prompt = current_prompt

        write_json(
            self.output_dir / f"stage3_merge_summary_iter_{iteration_idx:03d}.json",
            {
                "merge_history": merge_history,
                "final_consolidated_suggestion": final_suggestion,
                "rewritten_prompt": rewritten_prompt,
            },
        )
        return rewritten_prompt.strip(), merge_history

    @staticmethod
    def _format_group(group: list[str]) -> str:
        return "\n\n".join(f"Suggestion {idx + 1}:\n{item}" for idx, item in enumerate(group))
