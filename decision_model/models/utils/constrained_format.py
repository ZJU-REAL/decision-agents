"""Default score/choice prompts and answer-token probability readout.

No service addresses, model identities or HTTP calls belong in this module.
"""
from __future__ import annotations

import dataclasses
import math

from decision_model.models.utils.utils import render_score_input, render_choice_input, choice_options
from decision_model.types import DecisionResult, ChoiceOption

SCORE_SYSTEM_PROMPT = ('Answer the supplied question using only the supplied state. '
          'If asked for a probability, judge whether the described event occurs. '
          'Return exactly one token: Yes or No. '
          'Do not include spaces, explanations, punctuation, JSON, or Markdown.')


class ReadoutError(ValueError):
    pass


def encode_score(request, options):
    return dict(model=options['model'],
        messages=[dict(role='system', content=options.get('system_prompt', SCORE_SYSTEM_PROMPT)),
                  dict(role='user', content=render_score_input(request) + '\n\nAnswer:')],
        temperature=1.0, top_p=1.0, max_tokens=1,
        logprobs=True, top_logprobs=options.get('top_logprobs', 20))


def decode_score(request,envelope):
    choices=envelope.get('choices') or []
    if len(choices)!=1 or choices[0].get('finish_reason') not in ('length','stop'):
        raise ReadoutError('invalid_output_position')
    positions=(choices[0].get('logprobs') or {}).get('content') or []
    if len(positions)!=1:
        raise ReadoutError('missing_answer_logprobs')
    top=positions[0].get('top_logprobs') or []
    tokens=[t.get('token') for t in top]
    if len(tokens)!=len(set(tokens)):
        raise ReadoutError('duplicate_tokens')
    by={t['token']:t.get('logprob') for t in top}
    missing=[t for t in ['Yes','No'] if t not in by]
    if missing:
        raise ReadoutError('missing_binary_tokens:'+','.join(missing))
    a,b=by['Yes'],by['No']
    if any(type(p) not in (int,float) or not math.isfinite(p) or p>0 for p in [a,b]):
        raise ReadoutError('invalid_binary_logprobs')
    m=max(a,b);value=math.exp(a-m)/(math.exp(a-m)+math.exp(b-m))
    return DecisionResult.ok(request.request_id,value=value,text=choices[0].get('message',{}).get('content'),
        raw={'readout':dict(kind='constrained_score',labels=['Yes','No'],
            label_logprobs={'Yes':a,'No':b},label_probability_mass=math.exp(a)+math.exp(b),
            value=value,generated_token=positions[0].get('token'),
            normalization='exp(logp_Yes)/(exp(logp_Yes)+exp(logp_No))',
            score_semantics='Probability conditioned on the canonical Yes/No token set; not a calibration guarantee.')})


CHOICE_SYSTEM_PROMPT = (
    'Answer the supplied question using only the supplied state and available options. '
    'Each option ID is an opaque, case-sensitive label. '
    'Return exactly the ID of the best option as a single token. '
    'Do not include spaces, explanations, punctuation, JSON, or Markdown.'
)
ROUNDING_SLACK = 1e-6


def label_mapping(request, labels):
    choices = choice_options(request)
    if request.question.answer.kind != 'choice' or not 1 <= len(choices) <= len(labels):
        raise ReadoutError('unsupported_candidate_count_or_kind')
    return [dict(option_id=c.option_id, action=c.text, **label)
            for c, label in zip(choices, labels)]


def encode_choice(request, labels, options):
    mapping = label_mapping(request, labels)
    answer = dataclasses.replace(request.question.answer,
        choices=tuple(ChoiceOption(x['label'], x['action']) for x in mapping))
    relabeled = dataclasses.replace(request,
        question=dataclasses.replace(request.question, answer=answer))
    return dict(model=options['model'],
        messages=[dict(role='system', content=options.get('system_prompt', CHOICE_SYSTEM_PROMPT)),
                  dict(role='user', content=render_choice_input(relabeled) + '\n\nAnswer:')],
        temperature=1.0, top_p=1.0, max_tokens=1,
        logprobs=True, top_logprobs=options.get('top_logprobs', 20))


def decode_choice(request, envelope, labels, *, top_logprobs=20):
    """Argmax + conditional tail bound, assuming genuine global top-k logprobs."""
    mapping = label_mapping(request, labels)
    choices = envelope.get('choices') or []
    if len(choices) != 1:
        raise ReadoutError('invalid_choice_count')
    choice = choices[0]
    if choice.get('finish_reason') not in ('stop', 'length'):
        raise ReadoutError('invalid_finish_reason')
    positions = (choice.get('logprobs') or {}).get('content') or []
    if len(positions) != 1:
        raise ReadoutError('missing_single_answer_position')
    top = positions[0].get('top_logprobs') or []
    if len(top) not in (top_logprobs, top_logprobs + 1):
        raise ReadoutError('incomplete_global_top_logprobs')
    tokens = [t.get('token') for t in top]
    if any(not isinstance(t, str) for t in tokens) or len(set(tokens)) != len(top):
        raise ReadoutError('invalid_or_duplicate_tokens')
    logps = [t.get('logprob') for t in top]
    if any(type(p) not in (int, float) or not math.isfinite(p) or p > 0 for p in logps):
        raise ReadoutError('invalid_logprobs')
    # vLLM may put the sampled token first, even when it is not the argmax.
    # Exclude that entry when bounding the rank cutoff: it can lie outside top-k.
    # Remaining entries must form the ordered top-(k-1) prefix or longer.
    sampled = positions[0].get('token')
    ranked = [t for t in top if t['token'] != sampled]
    rank_logps = [t['logprob'] for t in ranked]
    if len(ranked) < top_logprobs - 1 or rank_logps != sorted(rank_logps, reverse=True):
        raise ReadoutError('invalid_ranked_prefix')
    by_label = dict(zip(tokens, logps))
    present = [x for x in mapping if x['label'] in by_label]
    if not present:
        raise ReadoutError('no_legal_label_in_top_logprobs')
    # List order resolves ties. Omitted labels are <= the final returned entry.
    best = max(present, key=lambda x: by_label[x['label']])
    if by_label[best['label']] < min(rank_logps):
        raise ReadoutError('unresolved_legal_argmax')
    masses = {x['option_id']: math.exp(by_label[x['label']]) for x in present}
    observed_mass = math.fsum(masses.values())
    total_top_mass = math.fsum(math.exp(p) for p in logps)
    if observed_mass <= 0 or total_top_mass > 1 + ROUNDING_SLACK:
        raise ReadoutError('invalid_probability_mass')
    missing = [x for x in mapping if x['label'] not in by_label]
    kth = math.exp(min(rank_logps))
    residual = min(1., max(0., 1. - total_top_mass) + ROUNDING_SLACK)
    omitted_upper = min(residual, len(missing) * (kth + ROUNDING_SLACK))
    error_upper = omitted_upper / (observed_mass + omitted_upper)
    readout = dict(
        kind='constrained_choice', top_logprobs=top_logprobs, label_mapping=mapping,
        candidate_count=len(mapping), observed_candidate_count=len(present),
        missing_option_ids=[x['option_id'] for x in missing],
        observed_option_logprobs={x['option_id']: by_label[x['label']] for x in present},
        observed_option_masses=masses,
        observed_option_probabilities={key: mass / observed_mass for key, mass in masses.items()},
        normalization="probabilities normalized over returned candidate labels",
        observed_candidate_mass=observed_mass, global_top_logprobs_mass=total_top_mass,
        kth_token_probability=kth, global_tail_mass_upper_bound=residual,
        rank_prefix_count=len(ranked), sampled_first=bool(tokens[0] == sampled),
        omitted_candidate_mass_upper_bound=omitted_upper,
        conditional_category_error_upper_bound=error_upper,
        rounding_slack=ROUNDING_SLACK,
        selected_option_id=best['option_id'], selected_label=best['label'],
        generated_token=positions[0].get('token'),
        argmax_tie_with_unreturned_possible=bool(missing and by_label[best['label']] == min(rank_logps)),
        assumptions=['returned entries include the global highest-probability prefix apart from the separately included sampled token, in one normalized next-token distribution',
                     'distinct single-token labels at the same first answer position'],
        probability_semantics='Conditioned on canonical candidate-label tokens; missing probabilities are unknown, not zero.',
    )
    return DecisionResult.ok(request.request_id, choice_id=best['option_id'],
                            text=best['label'], raw={'readout': readout})
