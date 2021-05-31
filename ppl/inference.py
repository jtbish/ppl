import abc
from collections import OrderedDict

NULL_ACTION = -1


class InferenceStrategyABC(metaclass=abc.ABCMeta):
    def __init__(self, action_space, default_action=NULL_ACTION):
        self._action_space = action_space
        if default_action != NULL_ACTION:
            assert default_action in self._action_space
        self._default_action = default_action

    @property
    def default_action(self):
        return self._default_action

    def __call__(self, clfrs, obs):
        action = self._infer_action(clfrs, obs)
        if self._default_action != NULL_ACTION and action == NULL_ACTION:
            action = self._default_action
        return action

    @abc.abstractmethod
    def _infer_action(self, clfrs, obs):
        raise NotImplementedError


class DecisionListInference(InferenceStrategyABC):
    def _infer_action(self, clfrs, obs):
        for clfr in clfrs:
            if clfr.does_match(obs):
                return clfr.action
        return NULL_ACTION


class SpecificityInference(InferenceStrategyABC):
    def _infer_action_(self, clfrs, obs):
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
            return NULL_ACTION
