import six
import sys
sys.modules['sklearn.externals.six'] = six
from mlrose import DiscreteOpt, TSPOpt, Queens, TravellingSales
import numpy as np

def queens_problem(state):
    fitness = Queens()

    return DiscreteOpt(
        length=state.shape[0], 
        fitness_fn=fitness, 
        maximize=False, 
        max_val=state.shape[0],
    ), fitness.evaluate(state)

def travelling_salesman_problem(state, seed):
    coords, length = [], state.shape[0]
    for i in range(length):
        coords.append((np.random.randint(length), np.random.randint(state.shape[0])))
    fitness = TravellingSales(coords=coords)

    return TSPOpt(
        length=length,
        fitness_fn=fitness,
        maximize=False
    ), fitness.evaluate(state)