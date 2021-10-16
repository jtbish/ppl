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
        self._hyperparams_dict = hyperparams_dict
        register_hyperparams(self._hyperparams_dict)
        seed_rng(get_hp("seed"))
        self._pop = None

    @property
    def pop(self):
        return self._pop

    def init(self):
        self._pop = init_pop(self._encoding, self._selectable_actions)
        self._assess_pop_perf(self._pop)
        return self._pop

    def run_gen(self):
        pop_size = get_hp("pop_size")
        assert (pop_size % 2) == 0
        num_breeding_rounds = (pop_size // 2)
        new_pop = []
        for _ in range(num_breeding_rounds):
            parent_a = copy.deepcopy(tournament_selection(self._pop))
            parent_b = copy.deepcopy(tournament_selection(self._pop))
            (child_a, child_b) = crossover(parent_a, parent_b, self._encoding)
            for child in (child_a, child_b):
                mutate(child, self._encoding, self._selectable_actions)
                new_pop.append(child)

        assert len(new_pop) == pop_size
        self._assess_pop_perf(new_pop)
        self._pop = new_pop
        return self._pop

    def _assess_pop_perf(self, pop):
        # process parallelism for perf assessment
        num_rollouts = get_hp("num_rollouts")
        gamma = get_hp("gamma")
        with Pool(_NUM_CPUS) as pool:
            results = pool.starmap(self._assess_indiv_perf,
                                   [(indiv, num_rollouts, gamma)
                                    for indiv in pop])
        for (indiv, result) in zip(pop, results):
            indiv.perf_assessment_res = result

    def _assess_indiv_perf(self, indiv, num_rollouts, gamma):
        return assess_perf(self._env, indiv, num_rollouts, gamma)
