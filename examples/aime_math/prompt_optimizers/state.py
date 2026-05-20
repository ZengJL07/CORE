from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GradientFeedback:
    text: str
    score: float
    context: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptVariable:
    value: str
    role_description: str
    requires_grad: bool = True
    gradients: list[GradientFeedback] = field(default_factory=list)

    def add_gradient(
        self,
        text: str,
        score: float,
        *,
        context: dict[str, Any] | None = None,
        **metadata: Any,
    ) -> None:
        self.gradients.append(
            GradientFeedback(
                text=text.strip(),
                score=score,
                context=dict(context) if context is not None else None,
                metadata=dict(metadata),
            )
        )

    def reset_gradients(self) -> None:
        self.gradients.clear()

    def set_value(self, value: str) -> None:
        self.value = value

    def get_role_description(self) -> str:
        return self.role_description

    def get_short_value(self, n_words_offset: int = 10) -> str:
        words = self.value.split(" ")
        if len(words) <= 2 * n_words_offset:
            return self.value
        return " ".join(words[:n_words_offset]) + " (...) " + " ".join(words[-n_words_offset:])

    def get_gradient_text(self) -> str:
        texts = [gradient.text for gradient in self.gradients if gradient.text.strip()]
        return "\n".join(texts)

    def get_gradient_and_context_text(self) -> str:
        parts: list[str] = []
        for gradient in self.gradients:
            if not gradient.text.strip():
                continue

            if gradient.context is None:
                parts.append(gradient.text)
                continue

            context = gradient.context
            parts.append(
                "\n".join(
                    [
                        "Here is a conversation:",
                        "",
                        "<CONVERSATION>",
                        f"Problem: {context.get('problem_input', '')}",
                        f"Model reasoning: {context.get('model_reasoning', '')}",
                        f"Model answer: {context.get('model_output', '')}",
                        "</CONVERSATION>",
                        "",
                        "This conversation is part of a larger system that solves math problems with a system prompt.",
                        f"The output is used as {context.get('response_desc', 'the final answer to the math problem')}.",
                        "",
                        f"Here is the feedback we got for {context.get('variable_desc', self.role_description)}:",
                        "",
                        f"<FEEDBACK>{gradient.text}</FEEDBACK>",
                        "",
                    ]
                )
            )
        return "\n".join(parts).strip()


@dataclass(frozen=True)
class PromptHistoryEntry:
    step_idx: int
    prompt: str
    batch_score: float
    val_score: float | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OptimizationStepArtifacts:
    prompt: str
    current_batch_score: float
    updated_batch_score: float
    combined_gradient: str
    metric_calls: int
    metadata: dict[str, Any] = field(default_factory=dict)
