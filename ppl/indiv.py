from .error import UnsetPropertyError


class Indiv:
    def __init__(self, clfrs, inference_strat):
        self._clfrs = list(clfrs)
        self._inference_strat = inference_strat
        self._fitness = None
        self._time_steps_used = None

    @property
    def classifiers(self):
        return self._clfrs

    @property
    def fitness(self):
        if self._fitness is None:
            raise UnsetPropertyError
        else:
            return self._fitness

    @fitness.setter
    def fitness(self, val):
        self._fitness = val

    @property
    def time_steps_used(self):
        """Time steps used on *most recent* fitness assessment."""
        if self._time_steps_used is None:
            raise UnsetPropertyError
        else:
            return self._time_steps_used

    @time_steps_used.setter
    def time_steps_used(self, val):
        self._time_steps_used = val

    def select_action(self, obs):
        """Performs inference on obs using classifiers to predict an action;
        i.e. making Indiv act as a policy."""
        return self._inference_strat(self._clfrs, obs)

    def __len__(self):
        return len(self._clfrs)
