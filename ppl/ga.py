from .condition import Condition
from .hyperparams import get_hyperparam as get_hp
from .indiv import Indiv
from .rng import get_rng


def tournament_selection(pop):
    def _select_random(pop):
        idx = get_rng().randint(0, len(pop))
        return pop[idx]

    best = _select_random(pop)
    for _ in range(2, get_hp("tourn_size") + 1):
        indiv = _select_random(pop)
        if indiv.fitness > best.fitness:
            best = indiv
    return best


def crossover(parent_a, parent_b, inference_strat):
    if get_rng().random() < get_hp("p_cross"):
        return _uniform_crossover(parent_a, parent_b, inference_strat)
    else:
        return (parent_a, parent_b)


def _two_point_crossover(parent_a, parent_b, inference_strat):
    """Two point crossover with cut points between classifiers."""
    n = get_hp("indiv_size")
    parent_a_clfrs = parent_a.classifiers
    parent_b_clfrs = parent_b.classifiers
    assert len(parent_a_clfrs) == n
    assert len(parent_b_clfrs) == n
    first = get_rng().randint(0, n + 1)
    second = get_rng().randint(0, n + 1)
    cut_start_idx = min(first, second)
    cut_end_idx = max(first, second)

    child_a_clfrs = parent_a_clfrs
    child_b_clfrs = parent_b_clfrs
    for idx in range(cut_start_idx, cut_end_idx):
        _swap(child_a_clfrs, child_b_clfrs, idx)

    assert len(child_a_clfrs) == n
    assert len(child_b_clfrs) == n
    child_a = Indiv(child_a_clfrs, inference_strat)
    child_b = Indiv(child_b_clfrs, inference_strat)
    return (child_a, child_b)


def _uniform_crossover(parent_a, parent_b, inference_strat):
    """Uniform crossover with cut points between classifiers."""
    n = get_hp("indiv_size")
    parent_a_clfrs = parent_a.classifiers
    parent_b_clfrs = parent_b.classifiers
    assert len(parent_a_clfrs) == n
    assert len(parent_b_clfrs) == n

    child_a_clfrs = parent_a_clfrs
    child_b_clfrs = parent_b_clfrs
    for idx in range(0, n):
        if get_rng().random() < get_hp("p_cross_swap"):
            _swap(child_a_clfrs, child_b_clfrs, idx)

    assert len(child_a_clfrs) == n
    assert len(child_b_clfrs) == n
    child_a = Indiv(child_a_clfrs, inference_strat)
    child_b = Indiv(child_b_clfrs, inference_strat)
    return (child_a, child_b)


def _swap(seq_a, seq_b, idx):
    seq_a[idx], seq_b[idx] = seq_b[idx], seq_a[idx]


def mutate(indiv, encoding, selectable_actions):
    """Mutates condition and action of clfrs contained within indiv by
    resetting them in Classifier object."""
    for clfr in indiv.classifiers:
        cond_alleles = clfr.condition.alleles
        mut_cond_alleles = encoding.mutate_condition_alleles(cond_alleles)
        mut_cond = Condition(mut_cond_alleles, encoding)
        mut_action = _mutate_action(clfr.action, selectable_actions)
        clfr.condition = mut_cond
        clfr.action = mut_action


def _mutate_action(action, selectable_actions):
    if get_rng().random() < get_hp("p_mut"):
        other_actions = list(set(selectable_actions) - {action})
        return get_rng().choice(other_actions)
    else:
        return action
