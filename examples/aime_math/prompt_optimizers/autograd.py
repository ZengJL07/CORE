from __future__ import annotations

from dataclasses import dataclass

from examples.aime_math.prompt_optimizers.state import PromptVariable


GRADIENT_TAG = "TEXT_GRADIENT"


def extract_tagged_text(response: str, tag: str) -> str:
    start_tag = f"<{tag}>"
    end_tag = f"</{tag}>"
    if start_tag in response and end_tag in response:
        return response.split(start_tag, 1)[1].split(end_tag, 1)[0].strip()
    return response.strip()


@dataclass(frozen=True)
class TextualLoss:
    prompt_variable: PromptVariable
    problem_input: str
    model_output: str
    model_reasoning: str
    score: float
    feedback: str

    def backward(self, backward_engine) -> None:
        if not self.prompt_variable.requires_grad:
            return

        prompt = "\n".join(
            [
                "You are writing a textual gradient to improve a math-solving system prompt.",
                f"Role of the variable: {self.prompt_variable.get_role_description()}",
                f"Current system prompt:\n{self.prompt_variable.value}",
                f"Problem:\n{self.problem_input}",
                f"Model reasoning:\n{self.model_reasoning}",
                f"Model answer:\n{self.model_output}",
                f"Metric score: {self.score:.2f}",
                f"Evaluation feedback:\n{self.feedback}",
                f"Return only the concrete improvement advice between <{GRADIENT_TAG}> and </{GRADIENT_TAG}>.",
            ]
        )
        response = backward_engine(prompt)
        gradient_text = extract_tagged_text(response, GRADIENT_TAG)
        self.prompt_variable.add_gradient(
            gradient_text,
            self.score,
            context={
                "problem_input": self.problem_input,
                "model_output": self.model_output,
                "model_reasoning": self.model_reasoning,
                "response_desc": "the final answer to the math problem",
                "variable_desc": self.prompt_variable.get_role_description(),
            },
        )


class TextualBatchLoss:
    def __init__(self, losses: list[TextualLoss]):
        self.losses = losses

    def backward(self, backward_engine) -> None:
        for loss in self.losses:
            loss.backward(backward_engine)


def build_textual_batch_loss(batch_evaluation, prompt_variable: PromptVariable) -> TextualBatchLoss:
    losses = [
        TextualLoss(
            prompt_variable=prompt_variable,
            problem_input=item.side_info["input"],
            model_output=item.side_info["output"],
            model_reasoning=item.side_info.get("reasoning", ""),
            score=item.score,
            feedback=item.side_info["execution_feedback"],
        )
        for item in batch_evaluation.example_evaluations
    ]
    return TextualBatchLoss(losses)
