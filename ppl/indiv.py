from .error import UnsetPropertyError
from .inference import infer_action


class Indiv:
    def __init__(self, rules):
        self._rules = list(rules)
        self._fitness = None
        self._time_steps_used = None

    @property
    def rules(self):
        return self._rules

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
        """Performs inference on obs using rules to predict an action;
        i.e. making Indiv act as a policy."""
        return infer_action(self, obs)

    def __len__(self):
        return len(self._rules)
