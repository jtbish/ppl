import abc

import numpy as np
from rlenvs.obs_space import IntegerObsSpace, RealObsSpace

from .hyperparams import get_hyperparam as get_hp
from .interval import Interval
from .rng import get_rng


class EncodingABC(metaclass=abc.ABCMeta):
    def __init__(self, obs_space):
        self._obs_space = obs_space

    @abc.abstractmethod
    def init_condition_alleles(self):
        raise NotImplementedError

    @abc.abstractmethod
    def decode(self, alleles):
        raise NotImplementedError

    @abc.abstractmethod
    def mutate_condition_alleles(self, cond_alleles):
        raise NotImplementedError


class UnorderedBoundEncodingABC(EncodingABC, metaclass=abc.ABCMeta):
    def init_condition_alleles(self):
        num_alleles = len(self._obs_space) * 2
        alleles = []
        for dim in self._obs_space:
            for _ in range(2):
                alleles.append(self._init_random_allele_for_dim(dim))
        assert len(alleles) == num_alleles
        return alleles

    @abc.abstractmethod
    def _init_random_allele_for_dim(self, dim):
        raise NotImplementedError

    def decode(self, alleles):
        phenotype = []
        assert len(alleles) % 2 == 0
        for i in range(0, len(alleles), 2):
            first_allele = alleles[i]
            second_allele = alleles[i + 1]
            lower = min(first_allele, second_allele)
            upper = max(first_allele, second_allele)
            phenotype.append(Interval(lower, upper))
        assert len(phenotype) == len(alleles) // 2
        return phenotype

    def mutate_condition_alleles(self, alleles):
        assert len(alleles) % 2 == 0
        allele_pairs = [(alleles[i], alleles[i + 1])
                        for i in range(0, len(alleles), 2)]
        mut_alleles = []
        for (allele_pair, dim) in zip(allele_pairs, self._obs_space):
            for allele in allele_pair:
                if get_rng().random() < get_hp("p_mut"):
                    noise = self._gen_mutation_noise()
                    sign = get_rng().choice([-1, 1])
                    mut_allele = allele + sign * noise
                    mut_allele = np.clip(mut_allele, dim.lower, dim.upper)
                    mut_alleles.append(mut_allele)
                else:
                    mut_alleles.append(allele)
        assert len(mut_alleles) == len(alleles)
        return mut_alleles

    @abc.abstractmethod
    def _gen_mutation_noise(self):
        raise NotImplementedError


def IntegerUnorderedBoundEncoding(UnorderedBoundEncodingABC):
    def __init__(self, obs_space):
        assert isinstance(obs_space, IntegerObsSpace)
        super().__init__(obs_space)

    def _init_random_allele_for_dim(self, dim):
        return get_rng().randint(low=dim.lower, high=(dim.upper + 1))

    def _gen_mutation_noise(self):
        # noise ~ [1, m_0]
        return get_rng().randint(low=1, high=(get_hp("m_nought") + 1))


def RealUnorderedBoundEncoding(UnorderedBoundEncodingABC):
    def __init__(self, obs_space):
        assert isinstance(obs_space, RealObsSpace)
        super().__init__(obs_space)

    def _init_random_allele_for_dim(self, dim):
        return get_rng().uniform(low=dim.lower, high=dim.upper)

    def _gen_mutation_noise(self):
        # noise ~ [0, m_0)
        return get_rng().uniform(low=0, high=get_hp("m_nought"))
