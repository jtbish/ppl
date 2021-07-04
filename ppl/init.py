from .classifier import Classifier
from .condition import Condition
from .hyperparams import get_hyperparam as get_hp
from .indiv import Indiv
from .rng import get_rng


def init_pop(encoding, selectable_actions, inference_strat):
    return [
        _init_indiv(encoding, selectable_actions, inference_strat)
        for _ in range(get_hp("pop_size"))
    ]


def _init_indiv(encoding, selectable_actions, inference_strat):
    num_clfrs = get_hp("indiv_size")
    clfrs = [
        _init_clfr(encoding, selectable_actions) for _ in range(num_clfrs)
    ]
    return Indiv(clfrs, inference_strat)


def _init_clfr(encoding, selectable_actions):
    condition = _init_clfr_condition(encoding)
    action = _init_clfr_action(selectable_actions)
    return Classifier(condition, action)


def _init_clfr_condition(encoding):
    return Condition(alleles=encoding.init_condition_alleles(),
                     encoding=encoding)


def _init_clfr_action(selectable_actions):
    return get_rng().choice(selectable_actions)
