from .error import UnsetFitnessError


class Indiv:
    def __init__(self, clfrs, inference_strat):
        self._clfrs = list(clfrs)
        self._inference_strat = inference_strat
        self._fitness = None

    @property
    def classifiers(self):
        return self._clfrs

    @property
    def fitness(self):
        if self._fitness is None:
            raise UnsetFitnessError
        else:
            return self._fitness

    @fitness.setter
    def fitness(self, val):
        self._fitness = val

    def select_action(self, obs):
        """Performs inference on obs using classifiers to predict an action;
        i.e. making Indiv act as a policy."""
        return self._inference_strat(self._clfrs, obs)

    def __len__(self):
        return len(self._clfrs)
