from .condition import Condition
from .hyperparams import get_hyperparam as get_hp
from .indiv import make_indiv
from .rng import get_rng
from .rule import Rule

_MIN_TOURN_SIZE = 2


def tournament_selection(pop):
    tourn_size = get_hp("tourn_size")
    assert tourn_size >= _MIN_TOURN_SIZE

    def _select_random(pop):
        idx = get_rng().randint(0, len(pop))
        return pop[idx]

    best = _select_random(pop)
    for _ in range(_MIN_TOURN_SIZE, (tourn_size + 1)):
        indiv = _select_random(pop)
        if indiv.fitness > best.fitness:
            best = indiv
    return best


def crossover(parent_a, parent_b, encoding):
    if get_rng().random() < get_hp("p_cross"):
        return _uniform_crossover_on_alleles(parent_a, parent_b, encoding)
    else:
        return (parent_a, parent_b)


def _uniform_crossover_on_rules(parent_a, parent_b):
    """Uniform crossover with swapping acting on whole rules within indivs."""
    num_rules = get_hp("indiv_size")

    assert len(parent_a.rules) == num_rules
    assert len(parent_b.rules) == num_rules
    child_a_rules = parent_a.rules
    child_b_rules = parent_b.rules

    for idx in range(0, num_rules):
        if get_rng().random() < get_hp("p_cross_swap"):
            _swap(child_a_rules, child_b_rules, idx)
    assert len(child_a_rules) == num_rules
    assert len(child_b_rules) == num_rules

    child_a = make_indiv(child_a_rules)
    child_b = make_indiv(child_b_rules)
    return (child_a, child_b)


def _uniform_crossover_on_alleles(parent_a, parent_b, encoding):
    """Uniform crossover with swapping acting on individual alleles of
    rules."""
    num_rules = get_hp("indiv_size")

    parent_a_alleles = []
    for rule in parent_a.rules:
        parent_a_alleles.extend(rule.condition.alleles)
        parent_a_alleles.append(rule.action)

    parent_b_alleles = []
    for rule in parent_b.rules:
        parent_b_alleles.extend(rule.condition.alleles)
        parent_b_alleles.append(rule.action)

    # 2 alleles for interval on each dim + action
    alleles_per_cond = (2 * len(encoding.obs_space))
    alleles_per_rule = (alleles_per_cond + 1)
    total_alleles = (num_rules * alleles_per_rule)
    assert len(parent_a_alleles) == total_alleles
    assert len(parent_b_alleles) == total_alleles

    child_a_alleles = parent_a_alleles
    child_b_alleles = parent_b_alleles
    for idx in range(0, total_alleles):
        if get_rng().random() < get_hp("p_cross_swap"):
            _swap(child_a_alleles, child_b_alleles, idx)

    child_a = _reassemble_child(child_a_alleles, encoding, alleles_per_rule,
                                alleles_per_cond, num_rules)
    child_b = _reassemble_child(child_b_alleles, encoding, alleles_per_rule,
                                alleles_per_cond, num_rules)
    return (child_a, child_b)


def _swap(seq_a, seq_b, idx):
    seq_a[idx], seq_b[idx] = seq_b[idx], seq_a[idx]


def _reassemble_child(alleles, encoding, alleles_per_rule, alleles_per_cond,
                      num_rules):
    rules = []
    cond_start_idxs = [(i * alleles_per_rule) for i in range(0, num_rules)]
    for cond_start_idx in cond_start_idxs:
        cond_end_idx_excl = (cond_start_idx + alleles_per_cond)
        cond_alleles = alleles[cond_start_idx:cond_end_idx_excl]
        action_idx = cond_end_idx_excl
        action = alleles[action_idx]
        cond = Condition(cond_alleles, encoding)
        rules.append(Rule(cond, action))
    assert len(rules) == num_rules
    return make_indiv(rules)


def mutate(indiv, encoding, selectable_actions):
    """Mutates condition and action of rules contained within indiv by
    resetting them in Rule object. Detects changes in genotype so as to mark
    indiv for performance evaluation."""
    genotype_changed = False
    for rule in indiv.rules:

        cond_alleles = rule.condition.alleles
        mut_cond_alleles = encoding.mutate_condition_alleles(cond_alleles)
        cond_alleles_changed = (mut_cond_alleles != cond_alleles)
        # only remake condition if alleles have changed
        if cond_alleles_changed:
            rule.condition = Condition(mut_cond_alleles, encoding)

        action = rule.action
        mut_action = _mutate_action(action, selectable_actions)
        action_changed = (mut_action != action)
        if action_changed:
            rule.action = mut_action

        rule_changed = (cond_alleles_changed or action_changed)
        genotype_changed = (genotype_changed or rule_changed)

    if genotype_changed:
        indiv.reinit()


def _mutate_action(action, selectable_actions):
    if get_rng().random() < get_hp("p_mut"):
        other_actions = list(set(selectable_actions) - {action})
        return get_rng().choice(other_actions)
    else:
        return action
