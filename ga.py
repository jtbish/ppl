import copy

from .condition import Condition
from .hyperparams import get_hyperparam as get_hp
from .indiv import Indiv
from .rng import get_rng


def tournament_selection(pop):
    def _select_random(pop):
        return get_rng().choice(pop)

    best = _select_random(pop)
    for _ in range(2, get_hp("tourn_size") + 1):
        indiv = _select_random(pop)
        if indiv.fitness > best.fitness:
            best = indiv
    return best


def crossover(parent_a, parent_b, inference_strat):
    indiv_size_min = get_hp("indiv_size_min")
    indiv_size_max = get_hp("indiv_size_max")

    # generate valid cut points for each indiv
    # to maintain closure (respect min and max size bounds), just select cut
    # points until closure is achieved
    cuts_are_valid = False
    while not cuts_are_valid:
        (a_cut_start_idx, a_cut_end_idx, a_cut_size) = \
            _select_cut_idxs(parent_a)
        (b_cut_start_idx, b_cut_end_idx, b_cut_size) = \
            _select_cut_idxs(parent_b)
        a_remainder_size = len(parent_a) - a_cut_size
        b_remainder_size = len(parent_b) - b_cut_size
        a_new_size = a_remainder_size + b_cut_size
        b_new_size = b_remainder_size + a_cut_size
        cuts_are_valid = (indiv_size_min <= a_new_size <= indiv_size_max) \
            and (indiv_size_min <= b_new_size <= indiv_size_max)

    # use the cuts to do the crossover
    # operate on copies of parent clfrs so refs are not shared
    parent_a_clfrs = copy.deepcopy(parent_a.classifiers)
    parent_b_clfrs = copy.deepcopy(parent_b.classifiers)

    a_cut = parent_a_clfrs[a_cut_start_idx:a_cut_end_idx]
    assert len(a_cut) == a_cut_size
    b_cut = parent_b_clfrs[b_cut_start_idx:b_cut_end_idx]
    assert len(b_cut) == b_cut_size

    child_a_clfrs = []
    child_a_clfrs.extend(parent_a_clfrs[0:a_cut_start_idx])
    child_a_clfrs.extend(b_cut)
    child_a_clfrs.extend(parent_a_clfrs[a_cut_end_idx:])
    assert indiv_size_min <= len(child_a_clfrs) <= indiv_size_max
    child_a = Indiv(child_a_clfrs, inference_strat)

    child_b_clfrs = []
    child_b_clfrs.extend(parent_b_clfrs[0:b_cut_start_idx])
    child_b_clfrs.extend(a_cut)
    child_b_clfrs.extend(parent_b_clfrs[b_cut_end_idx:])
    assert indiv_size_min <= len(child_b_clfrs) <= indiv_size_max
    child_b = Indiv(child_b_clfrs, inference_strat)

    return (child_a, child_b)


def _select_cut_idxs(indiv):
    # select two random cut idxs in indiv
    # for indiv of len n, there are n+1 cut idxs (beginning at 0: LHS of first
    # elem, ending at n: RHS of last elem)
    n = len(indiv)
    first = get_rng().choice(range(0, n + 1))
    second = get_rng().choice(range(0, n + 1))
    cut_start_idx = min(first, second)
    cut_end_idx = max(first, second)
    cut_size = (cut_end_idx - cut_start_idx)
    return (cut_start_idx, cut_end_idx, cut_size)


def mutate(indiv, encoding, action_space):
    """Mutates condition and action of clfrs contained within indiv by
    resetting them in Classifier object."""
    for clfr in indiv.classifiers:
        cond_alleles = clfr.condition.alleles
        mut_cond_alleles = encoding.mutate_condition_alleles(cond_alleles)
        mut_cond = Condition(mut_cond_alleles, encoding)
        mut_action = _mutate_action(clfr.action, action_space)
        clfr.condition = mut_cond
        clfr.action = mut_action


def _mutate_action(action, action_space):
    if get_rng().random() < get_hp("p_mut"):
        other_actions = list(set(action_space) - {action})
        return get_rng().choice(other_actions)
    else:
        return action
