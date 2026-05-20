from __future__ import annotations

from examples.aime_math.prompt_optimizers.autograd import extract_tagged_text


REFLECTION_TAG = "REFLECTION"
PROMPT_TAG = "IMPROVED_PROMPT"


def format_batch_feedback(batch_evaluation) -> str:
    parts: list[str] = []
    for idx, item in enumerate(batch_evaluation.example_evaluations, start=1):
        side_info = item.side_info
        if side_info.get("problem_id") is not None:
            failed_details = side_info.get("failed_details", []) or []
            first_failure = failed_details[0] if failed_details else {}
            parts.append(
                "\n".join(
                    [
                        f"Example {idx}",
                        f"Problem ID: {side_info.get('problem_id', '')}",
                        f"Task: {side_info.get('input', '')}",
                        f"Score: {item.score:.2f}",
                        f"Passed: {side_info.get('passed', False)}",
                        f"Generated code:\n{side_info.get('output', '')}",
                        f"Evaluation feedback: {side_info.get('execution_feedback', '')}",
                        f"First failing test: {first_failure.get('test', '')}",
                        f"Failure detail: {first_failure.get('error', '')}",
                        f"Stdout: {side_info.get('stdout', '')}",
                        f"Stderr: {side_info.get('stderr', '')}",
                        f"Traceback: {side_info.get('traceback', '')}",
                    ]
                ).strip()
            )
            continue
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


def _format_parent_context(parent_prompt: str | None, parent_reflection: str) -> str:
    if not (parent_prompt and parent_prompt.strip()) and not parent_reflection.strip():
        return ""

    lines = [
        "For additional context, below are the parent parameter value and the reflection that led to the current parameter value.",
        "Use this history to understand why the previous revision was made and why the current parameter took its present form.",
        "When using this history, pay special attention to:",
        "- Whether the previous reflection correctly identified the underlying issue",
        "- Whether the resulting changes were sufficient, especially if the current evaluation trajectory still seems related",
        "- Which ideas in the reflection remain useful and which may be overfit or misleading",
        "Use the parent parameter only to contextualize the reflection.",
        "When deciding what to carry forward, rely on the reflection's substantive ideas rather than the parent prompt's wording or structure.",
        "You may reorganize, revise, or drop earlier ideas if necessery, and you do not need to preserve the previous prompt structure.",
    ]

    if parent_prompt and parent_prompt.strip():
        lines.extend(
            [
                "Parent parameter value:",
                "```",
                parent_prompt.strip(),
                "```",
            ]
        )
    else:
        lines.append("No parent parameter value is available for the current parameter.")

    if parent_reflection.strip():
        lines.extend(
            [
                "Reflection associated with the transition from the parent parameter to the current one:",
                parent_reflection.strip(),
            ]
        )
    else:
        lines.append("No parent reflection is available for the current parameter.")

    return "\n".join(lines)


def build_reflection_prompt(
    *,
    current_prompt: str,
    current_batch_feedback: str,
    current_batch_score: float,
    parent_prompt: str | None,
    parent_reflection: str,
    include_parent_history: bool = True,
    branch_hint: str | None = None,
) -> str:
    del branch_hint  # Kept only for call-site compatibility.

    parent_context_block = (
        _format_parent_context(parent_prompt, parent_reflection)
        if include_parent_history
        else ""
    )

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

    if parent_context_block:
        sections.extend(["", parent_context_block])

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
