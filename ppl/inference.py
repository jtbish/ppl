NULL_ACTION = -1


def infer_action(indiv, obs):
    for rule in indiv.rules:
        if rule.does_match(obs):
            return rule.action
    return NULL_ACTION
