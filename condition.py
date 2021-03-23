_GENERALITY_LB_EXCL = 0.0
_GENERALITY_UB_INCL = 1.0


class Condition:
    def __init__(self, alleles, encoding):
        self._alleles = list(alleles)
        self._encoding = encoding
        self._phenotype = self._encoding.decode(self._alleles)

    @property
    def alleles(self):
        return self._alleles

    def does_match(self, obs):
        for (interval, obs_val) in zip(self._phenotype, obs):
            if not interval.contains_val(obs_val):
                return False
        return True

    def calc_generality(self, obs_space):
        # condition generality calc ala Wilson '00 Mining Oblique Data with XCS
        numer = sum([(interval.upper - interval.lower)
                     for interval in self._phenotype])
        denom = sum([(dim.upper - dim.lower) for dim in obs_space])
        generality = numer / denom
        assert _GENERALITY_LB_EXCL < generality <= _GENERALITY_UB_INCL
        return generality
