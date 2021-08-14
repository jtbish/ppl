from .classifier import Classifier
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


def crossover(parent_a, parent_b, inference_strat, encoding):
    if get_rng().random() < get_hp("p_cross"):
        return _uniform_crossover_v2(parent_a, parent_b, inference_strat,
                                     encoding)
    else:
        return (parent_a, parent_b)


def _two_point_crossover(parent_a, parent_b, inference_strat, encoding=None):
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


def _uniform_crossover(parent_a, parent_b, inference_strat, encoding=None):
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


def _uniform_crossover_v2(parent_a, parent_b, inference_strat, encoding):
    """Uniform crossover with cut points between individual genes."""
    n = get_hp("indiv_size")
    parent_a_clfrs = parent_a.classifiers
    parent_b_clfrs = parent_b.classifiers

    parent_a_alleles = []
    for clfr in parent_a_clfrs:
        parent_a_alleles.extend(clfr.condition.alleles)
        parent_a_alleles.append(clfr.action)

    parent_b_alleles = []
    for clfr in parent_b_clfrs:
        parent_b_alleles.extend(clfr.condition.alleles)
        parent_b_alleles.append(clfr.action)

    # 2 alleles for interval on each dim + action
    alleles_per_cond = 2*len(encoding.obs_space)
    alleles_per_clfr = alleles_per_cond + 1
    total_alleles = n * alleles_per_clfr
    assert len(parent_a_alleles) == total_alleles
    assert len(parent_b_alleles) == total_alleles

    child_a_alleles = parent_a_alleles
    child_b_alleles = parent_b_alleles
    for idx in range(0, total_alleles):
        if get_rng().random() < get_hp("p_cross_swap"):
            _swap(child_a_alleles, child_b_alleles, idx)

    def _reassemble_child(alleles, inference_strat, encoding):
        clfrs = []
        cond_start_idxs = [i*alleles_per_clfr for i in range(0, n)]
        for cond_start_idx in cond_start_idxs:
            cond_end_idx = (cond_start_idx + alleles_per_cond)
            cond_alleles = alleles[cond_start_idx:cond_end_idx]
            action = alleles[cond_end_idx]
            cond = Condition(cond_alleles, encoding)
            clfrs.append(Classifier(cond, action))
        assert len(clfrs) == n
        return Indiv(clfrs, inference_strat)

    child_a = _reassemble_child(child_a_alleles, inference_strat, encoding)
    child_b = _reassemble_child(child_b_alleles, inference_strat, encoding)
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
