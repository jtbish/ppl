import numpy as np

from .classifier import Classifier
from .condition import Condition
from .hyperparams import get_hyperparam as get_hp
from .indiv import Indiv


def init_pop(env, encoding, inference_strat):
    return [
        _init_indiv(env, encoding, inference_strat)
        for _ in range(get_hp("pop_size"))
    ]


def _init_indiv(env, encoding, inference_strat):
    num_clfrs = np.random.randint(low=get_hp("indiv_size_min"),
                                  high=get_hp("indiv_size_max") + 1)
    clfrs = [_init_clfr(env, encoding) for _ in range(num_clfrs)]
    return Indiv(clfrs, inference_strat)


def _init_clfr(env, encoding):
    condition = _init_clfr_condition(env.obs_space, encoding)
    action = _init_clfr_action(env.action_set)
    return Classifier(condition, action)


def _init_clfr_condition(obs_space, encoding):
    return Condition(alleles=encoding.init_condition_alleles(obs_space),
                     encoding=encoding)


def _init_clfr_action(action_set):
    return np.random.choice(list(action_set))
