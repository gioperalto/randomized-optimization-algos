import six
import sys
sys.modules['sklearn.externals.six'] = six
from mlrose import DiscreteOpt, Queens
import numpy as np

def queens_problem(state, seed):
    np.random.seed(seed)
    fitness = Queens()
    return DiscreteOpt(
        length=state.shape[0], 
        fitness_fn=fitness, 
        maximize=False, 
        max_val=state.shape[0]
    ), fitness.evaluate(state)
