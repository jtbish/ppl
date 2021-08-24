class Rule:
    def __init__(self, condition, action):
        self._condition = condition
        self._action = action

    @property
    def condition(self):
        return self._condition

    @condition.setter
    def condition(self, val):
        self._condition = val

    @property
    def action(self):
        return self._action

    @action.setter
    def action(self, val):
        self._action = val

    def does_match(self, obs):
        return self._condition.does_match(obs)

    def __str__(self):
        return f"{self._condition} -> {self._action}"
