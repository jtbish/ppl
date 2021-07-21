class Condition:
    def __init__(self, alleles, encoding):
        self._alleles = list(alleles)
        self._encoding = encoding
        # pre-compute the phenotype
        self._phenotype = self._encoding.decode(self._alleles)
#        self._generality = \
#            self._encoding.calc_condition_generality(self._phenotype)

    @property
    def alleles(self):
        return self._alleles

    @property
    def generality(self):
        return self._encoding.calc_condition_generality(self._phenotype)

    def does_match(self, obs):
        for (interval, obs_val) in zip(self._phenotype, obs):
            if not interval.contains_val(obs_val):
                return False
        return True

    def __str__(self):
        return " && ".join([str(interval) for interval in self._phenotype])
