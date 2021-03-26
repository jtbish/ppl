from .classifier import Classifier
from .condition import Condition
from .hyperparams import get_hyperparam as get_hp
from .indiv import Indiv
from .rng import get_rng


def init_pop(encoding, action_space, inference_strat):
    return [
        _init_indiv(encoding, action_space, inference_strat)
        for _ in range(get_hp("pop_size"))
    ]


def _init_indiv(encoding, action_space, inference_strat):
    num_clfrs = get_rng().randint(low=get_hp("indiv_size_min"),
                                  high=get_hp("indiv_size_max") + 1)
    clfrs = [_init_clfr(encoding, action_space) for _ in range(num_clfrs)]
    return Indiv(clfrs, inference_strat)


def _init_clfr(encoding, action_space):
    condition = _init_clfr_condition(encoding)
    action = _init_clfr_action(action_space)
    return Classifier(condition, action)


def _init_clfr_condition(encoding):
    return Condition(alleles=encoding.init_condition_alleles(),
                     encoding=encoding)


def _init_clfr_action(action_space):
    return get_rng().choice(action_space)
