"""Prompted-only JSON output instructions and response validation."""

from __future__ import annotations

import math
from typing import Any

from decision_model.types import DecisionRequest
from decision_model.models.utils.utils import (
    choice_options, finite_number, question_key, validate_score_spec,
)


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
