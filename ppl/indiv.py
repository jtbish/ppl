import abc

from .error import UnsetPropertyError
from .hyperparams import get_hyperparam as get_hp
from .inference import infer_action


def make_indiv(rules):
    use_policy_cache = get_hp("use_indiv_policy_cache")
    if use_policy_cache:
        return PolicyCacheIndiv(rules)
    else:
        return Indiv(rules)


class IndivABC(metaclass=abc.ABCMeta):
    def __init__(self, rules):
        self._rules = list(rules)
        self._perf_assessment_res = None

    @property
    def rules(self):
        return self._rules

    @property
    def perf_assessment_res(self):
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

    @abc.abstractmethod
    def reinit(self):
        """Used after any mutations occur in-place to mark that contents of
        Indiv object have changed without constructing new one."""
        raise NotImplementedError

    def select_action(self, obs):
        """Performs inference on obs using rules to predict an action;
        i.e. making Indiv act as a policy."""
        return infer_action(self, obs)

    def __len__(self):
        return len(self._rules)


class Indiv(IndivABC):
    def reinit(self):
        self._perf_assessment_res = None


class PolicyCacheIndiv(IndivABC):
    def __init__(self, rules):
        super().__init__(rules)
        self._policy_cache = {}

    def select_action(self, obs):
        obs_hashable = tuple(obs)
        try:
            return self._policy_cache[obs_hashable]
        except KeyError:
            action = infer_action(self, obs)
            self._policy_cache[obs_hashable] = action
            return action

    def reinit(self):
        self._perf_assessment_res = None
        self._policy_cache = {}
