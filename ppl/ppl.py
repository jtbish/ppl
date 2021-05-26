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
    def __init__(self, env, encoding, inference_strat, hyperparams_dict):
        self._env = env
        self._encoding = encoding
        self._inference_strat = inference_strat
        register_hyperparams(hyperparams_dict)
        seed_rng(get_hp("seed"))
        self._pop = None

    @property
    def pop(self):
        return self._pop

    def init(self):
        self._pop = init_pop(self._encoding, self._env.action_space,
                             self._inference_strat)
        self._eval_pop_fitness(self._pop)

    def run_gen(self):
        new_pop = []

        # elitism
        fitness_desc_sorted_pop = sorted(self._pop,
                                         key=lambda indiv: indiv.fitness,
                                         reverse=True)
        pop_size = get_hp("pop_size")
        num_elites = get_hp("num_elites")
        assert (pop_size - num_elites) % 2 == 0
        elites = fitness_desc_sorted_pop[0:num_elites]
        new_pop.extend(elites)

        # breeding
        num_breeding_rounds = (pop_size - num_elites) // 2
        for _ in range(0, num_breeding_rounds):
            parent_a = tournament_selection(self._pop)
            parent_b = tournament_selection(self._pop)
            (child_a, child_b) = crossover(parent_a, parent_b,
                                           self._inference_strat)
            mutate(child_a, self._encoding, self._env.action_space)
            mutate(child_b, self._encoding, self._env.action_space)
            new_pop.append(child_a)
            new_pop.append(child_b)
        assert len(new_pop) == pop_size

        # eval fitness
        self._pop = new_pop
        self._eval_pop_fitness(self._pop)

    def _eval_pop_fitness(self, pop):
        # process parallelism for fitness eval
        num_rollouts = get_hp("num_rollouts")
        gamma = get_hp("gamma")
        with Pool(_NUM_CPUS) as pool:
            results = pool.starmap(self._eval_indiv_fitness,
                                   [(indiv, num_rollouts, gamma)
                                    for indiv in pop])
        for (indiv, (expected_return, time_steps_used)) in zip(pop, results):
            indiv.fitness = expected_return
            indiv.time_steps_used = time_steps_used

    def _eval_indiv_fitness(self, indiv, num_rollouts, gamma):
        (expected_return,
         time_steps_used) = assess_perf(self._env,
                                        indiv,
                                        num_rollouts,
                                        gamma,
                                        return_time_steps_used=True)
        if expected_return is not None:
            return (expected_return, time_steps_used)
        else:
            return (self._env.perf_lower_bound, time_steps_used)
