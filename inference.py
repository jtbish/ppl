import abc
from .error import NoActionError


class IInferenceStrategy(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, clfrs, obs, obs_space=None):
        raise NotImplementedError


class DecisionListInference(IInferenceStrategy):
    def __call__(self, clfrs, obs, obs_space=None):
        for clfr in clfrs:
            if clfr.does_match(obs):
                return clfr.action
        raise NoActionError


class SpecificityInference(IInferenceStrategy):
    def __call__(self, clfrs, obs, obs_space):
        pass
