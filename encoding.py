import abc

import numpy as np

from .hyperparams import get_hyperparam as get_hp
from .interval import Interval


class IEncoding(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def init_condition_alleles(self, obs_space):
        raise NotImplementedError

    @abc.abstractmethod
    def decode(self, alleles):
        raise NotImplementedError

    @abc.abstractmethod
    def mutate_condition_alleles(self, cond_alleles, obs_space):
        raise NotImplementedError


class UnorderedBoundEncoding(IEncoding):
    def init_condition_alleles(self, obs_space):
        num_alleles = len(obs_space) * 2
        alleles = []
        for dim in obs_space:
            for _ in range(2):
                # choose bounds at random
                # TODO for now assume discrete only obs space, support cont. in
                # future
                alleles.append(
                    np.random.randint(low=dim.lower, high=(dim.upper + 1)))
        assert len(alleles) == num_alleles
        return alleles

    def decode(self, alleles):
        phenotype = []
        assert len(alleles) % 2 == 0
        for i in range(0, len(alleles), 2):
            first_idx = i
            second_idx = i + 1
            first_allele = alleles[first_idx]
            second_allele = alleles[second_idx]
            lower = min(first_allele, second_allele)
            upper = max(first_allele, second_allele)
            phenotype.append(Interval(lower, upper))
        assert len(phenotype) == len(alleles) // 2
        return phenotype

    def mutate_condition_alleles(self, alleles, obs_space):
        assert len(alleles) % 2 == 0
        allele_pairs = [(alleles[i], alleles[i + 1])
                        for i in range(0, len(alleles), 2)]
        mut_alleles = []
        for (allele_pair, dim) in zip(allele_pairs, obs_space):
            for allele in allele_pair:
                if np.random.random() < get_hp("p_mut"):
                    # TODO again assume discrete, add cont. in future
                    noise = np.random.randint(low=1,
                                              high=(get_hp("m_nought") + 1))
                    sign = np.random.choice([-1, 1])
                    mut_allele = allele + sign * noise
                    mut_allele = np.clip(mut_allele, dim.lower, dim.upper)
                    mut_alleles.append(mut_allele)
                else:
                    mut_alleles.append(allele)
        assert len(mut_alleles) == len(alleles)
        return mut_alleles
