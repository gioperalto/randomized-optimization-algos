import numpy as np
from util.plotter import Plotter
from util.algos import randomized_hill_climb, simulated_annealing, genetic_algorithm, mimic
from util.problems import knapsack_problem

def plot_rhc(fitness_fn, name, label):
    plt = Plotter(
        name=name,
        learner='optimization-problems/knapsack',
        axes={ 'x': 'Iterations', 'y': 'Fitness(x)' }
    )
    plt.add_plot(
        x=np.arange(1, len(rhc_fitness)+1, dtype=int), 
        y=rhc_fitness, 
        label=label, 
        marker=None
    )
    plt.save()

def plot_sa(problem, state, max_iters, name):
    temps = [.2, .4, .6, .8, 1.]

    plt = Plotter(
        name=name,
        learner='optimization-problems/knapsack',
        axes={ 'x': 'Iterations', 'y': 'Fitness(x)' },
        legend_title='Start temp'
    )
    for t in temps:
        sa_state, sa_score, sa_fitness = simulated_annealing(
            problem=problem,
            temp=t,
            state=state,
            max_iters=max_iters
        )
        plt.add_plot(
            x=np.arange(1, len(sa_fitness)+1, dtype=int),
            y=sa_fitness, 
            label=t, 
            marker=None
        )

    plt.save()

def plot_ga(problem, max_iters, name):
    mutate_probs = [.1, .3, .5]

    plt = Plotter(
        name=name,
        learner='optimization-problems/knapsack',
        axes={ 'x': 'Iterations', 'y': 'Fitness(x)' },
        legend_title='Mutation prob'
    )
    for mp in mutate_probs:
        ga_state, ga_score, ga_fitness = genetic_algorithm(
            problem=problem,
            max_iters=max_iters,
            mutation_prob=mp
        )
        plt.add_plot(
            x=np.arange(1, len(ga_fitness)+1, dtype=int),
            y=ga_fitness, 
            label=mp, 
            marker=None
        )

    plt.save()

def plot_mimic(problem, max_iters, name):
    keep_pcts = [.1, .3, .5, .7]

    plt = Plotter(
        name=name,
        learner='optimization-problems/knapsack',
        axes={ 'x': 'Iterations', 'y': 'Fitness(x)' },
        legend_title='Keep (%)'
    )
    for kp in keep_pcts:
        m_state, m_score, m_fitness = mimic(
            problem=problem,
            max_iters=max_iters,
            keep_pct=kp
        )
        plt.add_plot(
            x=np.arange(1, len(m_fitness)+1, dtype=int),
            y=m_fitness, 
            label=kp, 
            marker=None
        )

    plt.save()

if __name__ == "__main__":
    init_state = np.array([1, 0, 2, 1, 0])
    problem, init_score = knapsack_problem(state=init_state)

    rhc_state, rhc_score, rhc_fitness = randomized_hill_climb(
        problem=problem, 
        state=init_state, 
        max_iters=100
    )
    plot_rhc(
        fitness_fn=rhc_fitness, 
        name='Knapsack - RHC',
        label='Random Hill Climb'
    )

    print('RHC generated')

    plot_sa(
        problem=problem,
        state=init_state,
        max_iters=100,
        name='Knapsack - SA'
    )

    print('SA generated')

    plot_ga(
        problem=problem,
        max_iters=10,
        name='Knapsack - GA'
    )

    print('GA generated')

    plot_mimic(
        problem=problem,
        max_iters=10,
        name='Knapsack - MIMIC'
    )

    print('MIMIC generated')