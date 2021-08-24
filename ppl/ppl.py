import copy
import logging
import os
from multiprocessing import Pool

from rlenvs.environment import assess_perf

from .ga import crossover, mutate, tournament_selection
from .hyperparams import get_hyperparam as get_hp
from .hyperparams import register_hyperparams
from .init import init_pop
from .rng import seed_rng

_NUM_CPUS = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])


class PPL:
    def __init__(self, env, encoding, hyperparams_dict):
        self._env = env
        self._selectable_actions = self._env.action_space
        self._encoding = encoding
        register_hyperparams(hyperparams_dict)
        seed_rng(get_hp("seed"))
        self._pop = None

    @property
    def pop(self):
        return self._pop

    def init(self):
        self._pop = init_pop(self._encoding, self._selectable_actions)
        self._eval_pop_fitness(self._pop)
        return self._pop

    def run_gen(self):
        pop_size = get_hp("pop_size")
        assert pop_size % 2 == 0
        num_breeding_rounds = pop_size // 2
        new_pop = []
        for _ in range(num_breeding_rounds):
            parent_a = copy.deepcopy(tournament_selection(self._pop))
            parent_b = copy.deepcopy(tournament_selection(self._pop))
            (child_a, child_b) = crossover(parent_a, parent_b, self._encoding)
            for child in (child_a, child_b):
                mutate(child, self._encoding, self._selectable_actions)
                new_pop.append(child)

        assert len(new_pop) == pop_size
        self._eval_pop_fitness(new_pop)
        self._pop = new_pop
        return self._pop

    def _eval_pop_fitness(self, pop):
        # process parallelism for fitness eval
        num_rollouts = get_hp("num_rollouts")
        gamma = get_hp("gamma")
        with Pool(_NUM_CPUS) as pool:
            results = pool.starmap(self._eval_indiv_fitness,
                                   [(indiv, num_rollouts, gamma)
                                    for indiv in pop])
        for (indiv, result) in zip(pop, results):
            indiv.fitness = result.perf
            indiv.time_steps_used = result.time_steps_used

    def _eval_indiv_fitness(self, indiv, num_rollouts, gamma):
        return assess_perf(self._env, indiv, num_rollouts, gamma)
