from .ga import crossover, mutate, tournament_selection
from .hyperparams import get_hyperparam as get_hp
from .hyperparams import register_hyperparams
from .init import init_pop


class PPL:
    def __init__(self, env, encoding, inference_strat, hyperparams_dict):
        self._env = env
        self._encoding = encoding
        self._inference_strat = inference_strat
        register_hyperparams(hyperparams_dict)

        self._gen_counter = 0
        self._pop = None

    def init(self):
        self._pop = init_pop(self._env, self._encoding, self._inference_strat)
        self._eval_pop_fitness()
        return (self._gen_counter, self._pop)

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
            (child_a, child_b) = crossover(parent_a, parent_b, inference_strat)
            mutate(child_a, encoding, env)
            mutate(child_b, encoding, env)
            new_pop.append(child_a)
            new_pop.append(child_b)
        assert len(new_pop) == pop_size

        # eval fitness
        self._pop = new_pop
        self._eval_pop_fitness()
        return (self._gen_counter, self._pop)

    def _eval_pop_fitness(self):
        for indiv in self._pop:
            indiv.fitness = self._env.assess_perf(indiv)
