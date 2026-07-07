"""GEPA candidate proposer that reuses prompt_ucb's reflection prompt verbatim.

Upstream GEPA (``gepa.optimize_anything``) ships its own reflection meta-prompt
(``optimize_anything_reflection_prompt_template``) and its own ``<...>``-free,
single-``` fenced output format. prompt_ucb instead uses
``build_reflection_prompt`` with ``<REFLECTION>``/``<IMPROVED_PROMPT>`` tags and a
labeled feedback rendering.

To make the two paths send byte-identical content to the reflection model when
parent history is disabled, we plug this proposer into GEPA via
``ReflectionConfig.custom_candidate_proposer``. It:

1. Reconstructs ``(score, side_info)`` pairs from GEPA's reflective_dataset
   records (for AIME each record already carries the raw side_info keys,
   including ``score``).
2. Renders feedback and builds the reflection prompt through the SAME
   ``format_side_info_records`` / ``build_reflection_prompt`` used by prompt_ucb,
   with ``include_parent_history=False`` (GEPA has no parent-history notion).
3. Extracts the improved prompt from the ``<IMPROVED_PROMPT>`` tag, matching
   prompt_ucb's extractor.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from examples.aime_math.parent_reflection_gepa.reflection import (
    build_reflection_prompt,
    extract_reflection_and_prompt,
    format_side_info_records,
)


def _record_to_score_and_side_info(record: Mapping[str, Any]) -> tuple[float, dict]:
    """Turn one GEPA reflective_dataset record into a (score, side_info) pair.

    AIME's ``make_reflective_dataset`` passes the raw side_info through, so the
    record already contains ``score`` plus the other side_info keys. We rebuild a
    plain dict so ``format_side_info_records`` sees the exact same structure as
    prompt_ucb's ``item.side_info``.
    """
    side_info = dict(record)
    score = float(side_info.get("score", 0.0) or 0.0)
    return score, side_info


class GEPAAlignedReflectionProposer:
    """Callable matching ``gepa.core.adapter.ProposalFn``.

    Signature: ``(candidate, reflective_dataset, components_to_update) -> dict``.
    """

    def __init__(self, reflection_lm):
        self.reflection_lm = reflection_lm

    def __call__(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        proposed: dict[str, str] = {}
        for component_name in components_to_update:
            current_prompt = candidate[component_name]
            records = list(reflective_dataset.get(component_name, []))
            score_side_info = [_record_to_score_and_side_info(record) for record in records]

            feedback = format_side_info_records(score_side_info)
            batch_score = (
                sum(score for score, _ in score_side_info) / len(score_side_info)
                if score_side_info
                else 0.0
            )

            prompt = build_reflection_prompt(
                current_prompt=current_prompt,
                current_batch_feedback=feedback,
                current_batch_score=batch_score,
                parent_prompt=None,
                parent_reflection="",
                include_parent_history=False,
            )
            response = self.reflection_lm(prompt)
            _reflection, improved_prompt = extract_reflection_and_prompt(response)
            proposed[component_name] = improved_prompt if improved_prompt else current_prompt
        return proposed
