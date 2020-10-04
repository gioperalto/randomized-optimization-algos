import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose
import numpy as np

def randomized_hill_climb(problem, state, length):
    return mlrose.random_hill_climb(
        problem=problem, 
        init_state=state,
        max_attempts=length,
        # restarts=length
    )

def simulated_annealing(problem, state, length):
    schedule = mlrose.ExpDecay()
    return mlrose.simulated_annealing(
        problem=problem,
        schedule=schedule,
        max_attempts=length,
        init_state=state
    )

def genetic_algorithm(problem, length):
    return mlrose.genetic_alg(
        problem=problem,
        max_attempts=length
    )

def mimic(problem, length):
    return mlrose.mimic(
        problem=problem,
        max_attempts=length
    )

if __name__ == "__main__":
    np.random.seed(93) # Seeding
    init_state, fitness = np.arange(0, 8, 1, dtype=int), mlrose.Queens()
    problem = mlrose.DiscreteOpt(
        length=init_state.shape[0], 
        fitness_fn=fitness, 
        maximize=False, 
        max_val=init_state.shape[0]
    )

    print('Evaluate state: {}'.format((init_state, fitness.evaluate(init_state))))

    print(
        'Randomized hill climb:', 
        randomized_hill_climb(
            problem=problem, 
            state=init_state, 
            length=init_state.shape[0]
        )
    )
    print(
        'Simulated annealing:',
        simulated_annealing(
            problem=problem,
            state=init_state,
            length=init_state.shape[0]
        )
    )
    print(
        'Genetic algorithm:',
        genetic_algorithm(
            problem=problem,
            length=init_state.shape[0]
        )
    )
    print(
        'MIMIC:',
        mimic(
            problem=problem,
            length=init_state.shape[0]
        )
    )