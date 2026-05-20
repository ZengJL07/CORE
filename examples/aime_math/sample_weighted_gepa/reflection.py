from __future__ import annotations

from examples.aime_math.prompt_optimizers.autograd import extract_tagged_text
from examples.aime_math.sample_weighted_gepa.state import InnerLoopAttempt


REFLECTION_TAG = "REFLECTION"
PROMPT_TAG = "IMPROVED_PROMPT"


def format_batch_feedback(batch_evaluation) -> str:
    parts: list[str] = []
    for idx, item in enumerate(batch_evaluation.example_evaluations, start=1):
        side_info = item.side_info
        parts.append(
            "\n".join(
                [
                    f"Example {idx}",
                    f"Problem: {side_info.get('input', '')}",
                    f"Score: {item.score:.2f}",
                    f"Model reasoning: {side_info.get('reasoning', '')}",
                    f"Model answer: {side_info.get('output', '')}",
                    f"Evaluation feedback: {side_info.get('execution_feedback', '')}",
                ]
            ).strip()
        )
    return "\n\n".join(parts)


def _format_previous_attempts(previous_attempts: list[InnerLoopAttempt]) -> str:
    if len(previous_attempts) <= 1:
        return ""

    sections: list[str] = []
    for attempt in previous_attempts[1:]:
        lines = [
            f"Previous optimization step {attempt.step_idx}:",
            "Prompt produced in that step:",
            "```",
            attempt.prompt.strip(),
            "```",
        ]
        if attempt.reflection.strip():
            lines.extend(
                [
                    "Reflection associated with that step:",
                    attempt.reflection.strip(),
                ]
            )
        if attempt.train_score == attempt.train_score:
            lines.append(f"Observed minibatch score in that step: {attempt.train_score:.2f}")
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def build_reflection_prompt(
    *,
    current_prompt: str,
    current_batch_feedback: str,
    current_batch_score: float,
    previous_attempts: list[InnerLoopAttempt],
    branch_hint: str | None = None,
) -> str:
    del branch_hint  # Kept only for call-site compatibility.

    previous_attempts_block = _format_previous_attempts(previous_attempts)

    sections = [
        "I am optimizing a parameter in my system. This parameter may be a literal value or the text prompt/instruction used to guide how problems are solved. The current parameter value is:",
        "```",
        current_prompt.strip(),
        "```",
        "",
        "Below is evaluation data showing how this parameter value performed across multiple test cases. The data contains performance metrics, diagnostic information, and other relevant details from the evaluation:",
        "```",
        f"Current minibatch average score: {current_batch_score:.2f}",
        "",
        current_batch_feedback.strip(),
        "```",
    ]

    if previous_attempts_block:
        sections.extend(
            [
                "",
                "For additional context, below are the results of earlier optimization steps from this same inner-loop search. These are previous optimization results provided only as reference. Use them to avoid repeating unhelpful edits, but do not mechanically copy them.",
                previous_attempts_block,
            ]
        )

    sections.extend(
        [
            "",
            "Your task is to propose a new, improved parameter value, or improved problem-solving prompt text, that can be used as a drop-in replacement for the current one.",
            "",
            "Carefully analyze all the evaluation data provided above. Look for patterns that indicate what works and what doesn't. Pay special attention to:",
            "- Performance metrics and how they correlate with parameter behavior",
            "- Recurring issues, errors, or failure patterns across multiple test cases",
            "- Successful patterns or behaviors that should be preserved or enhanced",
            "- Any domain-specific requirements, constraints, or factual information revealed in the evaluation data",
            "- Specific technical details that are crucial for understanding the parameter's role",
            "",
            "",
            "Based on your analysis, propose a new parameter value or prompt text that addresses the identified issues while maintaining or improving upon what works well. Your proposal should be directly informed by the patterns and insights from the evaluation data.",
            "",
            f"Write the summary of your analysis above between <{REFLECTION_TAG}> and </{REFLECTION_TAG}>. After that, write the final new parameter value or prompt text between <{PROMPT_TAG}> and </{PROMPT_TAG}>.",
            "",
            f"Provide the your reflection within exactly one <{REFLECTION_TAG}> followed by exactly one <{PROMPT_TAG}> block.",
        ]
    )

    return "\n".join(sections).strip()


def extract_reflection_and_prompt(response: str) -> tuple[str, str]:
    reflection = extract_tagged_text(response, REFLECTION_TAG)
    prompt = extract_tagged_text(response, PROMPT_TAG)
    return reflection.strip(), prompt.strip()
