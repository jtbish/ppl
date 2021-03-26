class Interval:
    def __init__(self, lower, upper):
        assert lower <= upper
        self._lower = lower
        self._upper = upper

    @property
    def lower(self):
        return self._lower

    @property
    def upper(self):
        return self._upper

    def contains_val(self, val):
        return self._lower <= val <= self._upper

    def __str__(self):
        return f"[{self._lower}, {self._upper}]"
