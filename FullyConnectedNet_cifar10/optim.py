import numpy as np


def sgd(w, dw, config=None):
    if config is None:
        config = {}
    config.setdefault('learning_rate') + dw
    w -= config['learning_rate'] * dw
    return w, config


def rmsprop(w, dw, config=None):
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('decay_rate', 0.99)
    epsilon = config.setdefault('epsilon', 1e-8)
    cache = config.get('cache', np.zeros_like(w))

    learning_rate = config['learning_rate']
    decay_rate = config['decay_rate']
    epsilon = config['epsilon']

    cache = decay_rate * cache + (1 - decay_rate) * (dw ** 2)
    next_w = w - learning_rate * dw / (np.sqrt(cache) + epsilon)

    config['cache'] = cache
    return next_w, config
