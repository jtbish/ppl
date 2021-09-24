from .error import UnsetPropertyError
from .inference import infer_action


class Indiv:
    def __init__(self, rules):
        self._rules = list(rules)
        # *most recent* perf assessment result
        self._perf_assessment_res = None

    @property
    def rules(self):
        return self._rules

    @property
    def perf_assessment_res(self):
        if self._perf_assessment_res is None:
            raise UnsetPropertyError
        else:
            return self._perf_assessment_res

    @perf_assessment_res.setter
    def perf_assessment_res(self, val):
        self._perf_assessment_res = val

    @property
    def fitness(self):
        if self._perf_assessment_res is None:
            raise UnsetPropertyError
        else:
            # fitness == perf
            return self._perf_assessment_res.perf

    def select_action(self, obs):
        """Performs inference on obs using rules to predict an action;
        i.e. making Indiv act as a policy."""
        return infer_action(self, obs)

    def __len__(self):
        return len(self._rules)
