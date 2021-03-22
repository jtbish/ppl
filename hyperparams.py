# mask access to registry behind accessor functions to ensure callers can't
# mutate it by accident - it is "immutable"
_hyperparams_registry = {}


def register_hyperparams(hyperparams_dict):
    global _hyperparams_registry
    _hyperparams_registry = {**_hyperparams_registry, **hyperparams_dict}


def get_hyperparam(name):
    return _hyperparams_registry[name]
