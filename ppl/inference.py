import abc
from collections import OrderedDict

from .error import NoActionError


class InferenceStrategyABC(metaclass=abc.ABCMeta):
    def __init__(self, action_space):
        self._action_space = action_space

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
            for action in self._action_space:
                action_set = [
                    clfr for clfr in match_set if clfr.action == action
                ]
                if len(action_set) > 0:
                    min_generality_as = min(
                        [clfr.calc_generality() for clfr in action_set])
                    generality_map[action] = min_generality_as
            most_specific_action = min(generality_map, key=generality_map.get)
            return most_specific_action
        else:
            raise NoActionError
