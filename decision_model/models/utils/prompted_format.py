"""Prompted JSON answer formats: instructions, input rendering, and validation.

Score and choice share these formats across generated-answer clients. Task
questions and context arrive in DecisionRequest; no service calls occur here.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, Tuple

from decision_model.types import ChoiceOption, DecisionRequest
from decision_model.models.utils.utils import finite_number, question_key


GENERATED_CHOICE_SYSTEM_PROMPT = """You select the next action for an agent.

Answer the question using only the supplied state and available options.

Assign a probability to every option, expressing your estimate that
it is the best next action under the supplied question.
These probabilities describe the relative preference among the options,
not the probability of eventual task success.

Return a single JSON object with this structure:
{
  "answers": {
    "<question_key>": {
      "type": "choice",
      "choice": "<selected_option_id>",
      "probabilities": {
        "<option_id>": 0.0
      },
      "confidence": 0.0
    }
  }
}

Use the supplied question key and option IDs exactly.
Include every option ID exactly once in probabilities.
Each probability must be between 0 and 1, and their sum must equal 1.
Select the option with the highest probability. Break ties by choosing
the option appearing first in the supplied options.
Set confidence to the selected option's probability.

Do not include additional fields, explanations, or Markdown fences."""



GENERATED_SCORE_SYSTEM_PROMPT = """Answer the supplied question with a probability.

Use only the supplied state, question and answer instructions.

Return a single JSON object with this structure:
{
  "answers": {
    "<question_key>": {
      "type": "score",
      "value": 0.0
    }
  }
}

Use the supplied question key exactly.
Set value to a finite JSON number between 0 and 1, inclusive.
Include only the supplied question key in answers.
Do not include additional fields, explanations, or Markdown fences."""



PROBABILITY_TOLERANCE = 1e-6



def choice_options(request: DecisionRequest) -> Tuple[ChoiceOption, ...]:
    if request.question.answer.kind != "choice":
        raise ValueError("generated choice supports choice answers only")
    options = request.question.answer.choices
    if not 1 <= len(options) <= 255:
        raise ValueError("unsupported_candidate_count")
    if any(not isinstance(option, ChoiceOption)
           or not isinstance(option.option_id, str) or not option.option_id.strip()
           or not isinstance(option.text, str) or not option.text.strip() for option in options):
        raise ValueError("choice options require non-empty IDs and text")
    if len({option.option_id for option in options}) != len(options):
        raise ValueError("choice option IDs must be unique")
    return options



def render_choice_input(request: DecisionRequest) -> str:
    """Render only the visible state, question and unchanged ordered options."""
    options = choice_options(request)
    question = request.question
    key = question_key(request)
    if not isinstance(question.question, str) or not question.question.strip():
        raise ValueError("question must be a non-empty string")
    if not isinstance(question.answer.instructions, str):
        raise ValueError("answer instructions must be a string")
    instructions = question.question
    if question.answer.instructions:
        instructions += "\n\n" + question.answer.instructions
    if not isinstance(request.context.text, str):
        raise ValueError("context text must be a string")
    if "state" in request.context.materials:
        if request.context.text:
            raise ValueError("supply structured state or context text, not both")
        state = request.context.materials["state"]
    else:
        state = request.context.text
    content = "Question key: {}\n\nState:\n{}\n\nQuestion:\n{}\n\nAvailable options:\n{}".format(
        key,
        json.dumps(state, ensure_ascii=False, allow_nan=False),
        instructions,
        json.dumps([{"id": option.option_id, "action": option.text} for option in options],
                   ensure_ascii=False, allow_nan=False),
    )
    return content



def validate_score_spec(request: DecisionRequest) -> None:
    """Require a score on the numeric unit interval without choice options."""
    answer = request.question.answer
    if answer.kind != "score":
        raise ValueError("generated score supports score answers only")
    if (len(answer.scale) != 2 or not all(finite_number(bound) for bound in answer.scale)
            or answer.scale[0] != 0 or answer.scale[1] != 1):
        raise ValueError("score scale must be the finite numeric interval (0, 1)")
    if answer.choices:
        raise ValueError("score answers must not supply choice options")



def render_score_input(request: DecisionRequest) -> str:
    """Render only the visible state, original question and answer instructions."""
    validate_score_spec(request)
    question = request.question
    key = question_key(request)
    if not isinstance(question.question, str) or not question.question.strip():
        raise ValueError("question must be a non-empty string")
    if not isinstance(question.answer.instructions, str):
        raise ValueError("answer instructions must be a string")
    instructions = question.question
    if question.answer.instructions:
        instructions += "\n\n" + question.answer.instructions
    if not isinstance(request.context.text, str):
        raise ValueError("context text must be a string")
    if "state" in request.context.materials:
        if request.context.text:
            raise ValueError("supply structured state or context text, not both")
        state = request.context.materials["state"]
    else:
        state = request.context.text
    return "Question key: {}\n\nState:\n{}\n\nQuestion:\n{}".format(
        key, json.dumps(state, ensure_ascii=False, allow_nan=False), instructions,
    )



def validate_generated_score(request: DecisionRequest, response: Any) -> float:
    """Validate a decoded generated-score answer without repair or tolerance."""
    try:
        validate_score_spec(request)
        key = question_key(request)
        if not isinstance(response, dict) or set(response) != {"answers"}:
            raise ValueError("invalid answer envelope")
        answers = response["answers"]
        if not isinstance(answers, dict) or set(answers) != {key}:
            raise ValueError("invalid question key coverage")
        answer = answers[key]
        if not isinstance(answer, dict) or set(answer) != {"type", "value"}:
            raise ValueError("invalid answer fields")
        value = answer["value"]
        if answer["type"] != "score" or not finite_number(value) or not 0 <= value <= 1:
            raise ValueError("invalid score value")
        return float(value)
    except (TypeError, OverflowError, AttributeError, RecursionError):
        raise ValueError("invalid generated score response") from None



def validate_generated_choice(request: DecisionRequest, response: Any) -> str:
    """Validate a decoded generated-choice answer without repairing it."""
    try:
        options = choice_options(request)
        key = question_key(request)
        if not isinstance(response, dict) or set(response) != {"answers"}:
            raise ValueError("invalid answer envelope")
        answers = response["answers"]
        if not isinstance(answers, dict) or set(answers) != {key}:
            raise ValueError("invalid question key coverage")
        answer = answers[key]
        if not isinstance(answer, dict) or set(answer) != {"type", "choice", "probabilities", "confidence"}:
            raise ValueError("invalid answer fields")
        choice = answer["choice"]
        probabilities = answer["probabilities"]
        identifiers = [option.option_id for option in options]
        if answer["type"] != "choice" or not isinstance(choice, str) or choice not in identifiers:
            raise ValueError("invalid choice ID")
        if not isinstance(probabilities, dict) or set(probabilities) != set(identifiers):
            raise ValueError("invalid probability coverage")
        if any(not finite_number(p) or not 0 <= p <= 1 for p in probabilities.values()):
            raise ValueError("invalid probability")
        if choice != max(identifiers, key=probabilities.__getitem__):
            raise ValueError("choice must maximize probability with input-order ties")
        confidence = answer["confidence"]
        if (not finite_number(confidence) or not 0 <= confidence <= 1
                or not math.isclose(confidence, probabilities[choice],
                                    rel_tol=0.0, abs_tol=PROBABILITY_TOLERANCE)):
            raise ValueError("confidence must equal the selected probability")
        return choice
    except (TypeError, OverflowError, AttributeError, RecursionError):
        raise ValueError("invalid generated choice response") from None
