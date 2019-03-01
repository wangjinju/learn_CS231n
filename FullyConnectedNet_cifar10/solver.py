from fullyConnectedNet import FullyConnectedNet
import numpy as np
import matplotlib.pyplot as plt
import optim


class Solver(object):
    '''
    solver = Solver(model, data,
                optim_config={
                    'learning_rate': learning_rates[uo]
                },
                batch_size=100,
                iters_per_ann=400,
                num_epochs=5,
                update_rule=uo,
                print_every=100,
                verbose=True,
                lr_decay=1
                )
    '''

    def __init__(self, model, data, **kwargs):

        pass

    def _step(self):
        pass

    def check_accuracy(self, X, y, num_samples=None):
        pass

    def train(self):
        pass

    def visualization_model(self):
        pass
    pass
