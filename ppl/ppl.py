import copy
import logging
import os
from multiprocessing import Pool

from rlenvs.environment import assess_perf

from .ga import crossover, mutate, tournament_selection
from .hyperparams import get_hyperparam as get_hp
from .hyperparams import register_hyperparams
from .inference import NULL_ACTION
from .init import init_pop
from .rng import seed_rng

_NUM_CPUS = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])


class PPL:
    def __init__(self, env, encoding, inference_strat, hyperparams_dict):
        self._env = env
        self._encoding = encoding
        self._inference_strat = inference_strat
        self._selectable_actions = \
            self._find_selectable_actions(self._env.action_space,
                                          self._inference_strat.default_action)
        register_hyperparams(hyperparams_dict)
        seed_rng(get_hp("seed"))
        self._pop = None

    @property
    def pop(self):
        return self._pop

    @property
    def selectable_actions(self):
        return self._selectable_actions

    def _find_selectable_actions(self, action_space, default_action):
        """Selectable actions are everything but default."""
        if default_action != NULL_ACTION:
            return tuple(set(action_space) - {default_action})
        else:
            return action_space

    def init(self):
        self._pop = init_pop(self._encoding, self._selectable_actions,
                             self._inference_strat)
        self._eval_indivs_fitness(self._pop)
        return self._pop

    def run_gen(self):
        new_pop = []

        # elitism
        # first mark everyone as non-elite
        for indiv in self._pop:
            indiv.is_elite = False
        # then do elitist selection
        fitness_desc_sorted_pop = sorted(self._pop,
                                         key=lambda indiv: indiv.fitness,
                                         reverse=True)
        pop_size = get_hp("pop_size")
        num_elites = get_hp("num_elites")
        assert (pop_size - num_elites) % 2 == 0
        # make deepcopies of elites because they could possibly be selected as
        # parents later and modified, don't want to share refs for that
        elites = []
        for idx in range(0, num_elites):
            elite = fitness_desc_sorted_pop[idx]
            elite.is_elite = True
            elites.append(elite)

        # breed offspring
        num_breeding_rounds = (pop_size - num_elites) // 2
        offspring = []
        for _ in range(0, num_breeding_rounds):
            # take parents as copies
            parent_a = copy.deepcopy(tournament_selection(self._pop))
            parent_b = copy.deepcopy(tournament_selection(self._pop))
            (child_a, child_b) = crossover(parent_a, parent_b,
                                           self._inference_strat)
            mutate(child_a, self._encoding, self._selectable_actions)
            mutate(child_b, self._encoding, self._selectable_actions)
            offspring.append(child_a)
            offspring.append(child_b)
        self._eval_indivs_fitness(offspring)

        # build new pop
        new_pop = elites + offspring
        assert len(new_pop) == pop_size
        self._pop = new_pop
        return self._pop

    def _eval_indivs_fitness(self, indivs):
        # process parallelism for fitness eval
        num_rollouts = get_hp("num_rollouts")
        gamma = get_hp("gamma")
        with Pool(_NUM_CPUS) as pool:
            results = pool.starmap(self._eval_indiv_fitness,
                                   [(indiv, num_rollouts, gamma)
                                    for indiv in indivs])
        for (indiv, result) in zip(indivs, results):
            indiv.fitness = result.perf
            indiv.time_steps_used = result.time_steps_used

    def _eval_indiv_fitness(self, indiv, num_rollouts, gamma):
        return assess_perf(self._env, indiv, num_rollouts, gamma)
