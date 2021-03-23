import abc
from collections import OrderedDict

from .error import NoActionError


class InferenceStrategyABC(metaclass=abc.ABCMeta):
    def __init__(self, env):
        self._env = env

    @abc.abstractmethod
    def __call__(self, clfrs, obs):
        raise NotImplementedError


class DecisionListInference(InferenceStrategyABC):
    def __call__(self, clfrs, obs):
        for clfr in clfrs:
            if clfr.does_match(obs):
                return clfr.action
        raise NoActionError


class SpecificityInference(InferenceStrategyABC):
    def __call__(self, clfrs, obs):
        match_set = [clfr for clfr in clfrs if clfr.does_match(obs)]
        if len(match_set) > 0:
            generality_map = OrderedDict()
            for action in self._env.action_space:
                action_set = [
                    clfr for clfr in match_set if clfr.action == action
                ]
                if len(action_set) > 0:
                    min_generality_as = min([
                        clfr.calc_generality(self._env.obs_space)
                        for clfr in action_set
                    ])
                    generality_map[action] = min_generality_as
            most_specific_action = min(generality_map, key=generality_map.get)
            return most_specific_action
        else:
            raise NoActionError
