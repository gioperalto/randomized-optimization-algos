import six
import sys
sys.modules['sklearn.externals.six'] = six
from mlrose import DiscreteOpt, TSPOpt, Queens, TravellingSales, Knapsack
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

def knapsack_problem(state):
    length = state.shape[0]
    weights, values = [10, 5, 2, 8, 15], [1, 2, 3, 4, 5]
    fitness = Knapsack(weights, values, max_weight_pct=.6)

    return DiscreteOpt(
        length=length,
        fitness_fn=fitness,
        maximize=True,
        max_val=length
    ), fitness.evaluate(state)