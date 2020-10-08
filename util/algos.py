import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose
import numpy as np

def randomized_hill_climb(problem, state, max_iters):
    np.random.seed(93)
    return mlrose.random_hill_climb(
        problem=problem, 
        init_state=state,
        max_attempts=max_iters,
        max_iters=max_iters,
        curve=True,
        random_state=1
    )

def simulated_annealing(problem, temp, state, max_iters):
    np.random.seed(93)
    schedule = mlrose.GeomDecay(init_temp=temp)
    return mlrose.simulated_annealing(
        problem=problem,
        schedule=schedule,
        max_attempts=max_iters,
        max_iters=max_iters,
        init_state=state,
        curve=True,
        random_state=1
    )

def genetic_algorithm(problem, max_iters, mutation_prob=0.1):
    np.random.seed(93)
    return mlrose.genetic_alg(
        problem=problem,
        max_attempts=max_iters,
        max_iters=max_iters,
        mutation_prob=mutation_prob,
        curve=True,
        random_state=1
    )

def mimic(problem, max_iters, keep_pct=0.2):
    np.random.seed(93)
    return mlrose.mimic(
        problem=problem,
        max_attempts=max_iters,
        max_iters=max_iters,
        keep_pct=keep_pct,
        curve=True,
        random_state=1
    )