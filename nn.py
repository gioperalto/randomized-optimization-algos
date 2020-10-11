import six
import sys
sys.modules['sklearn.externals.six'] = six
from util.train_test import heart_failure_prediction, scale_data, encode_data
from util.plotter import Plotter
from sklearn.neural_network import MLPClassifier
from mlrose import NeuralNetwork, GeomDecay
from tqdm import tqdm
import warnings
import numpy as np
import time

def warn(*args, **kwargs):
    pass

def create_scikit_nn(x, y, max_iter):
    return MLPClassifier(
        solver='sgd',
        random_state=0,
        hidden_layer_sizes=(x.shape[1]**2, x.shape[1], x.shape[1]**2),
        max_iter=max_iter,
        n_iter_no_change=1
    ).fit(x, y)

def create_sgd_nn(x, y, max_iter, lr):
    return NeuralNetwork(
        hidden_nodes=[x.shape[1], x.shape[1]],
        algorithm='gradient_descent',
        max_iters=max_iter,
        learning_rate=lr,
        random_state=0,
        curve=True,
    ).fit(x, y)

def create_rhc_nn(x, y, max_iter, restarts):
    return NeuralNetwork(
        hidden_nodes=[x.shape[1], x.shape[1]],
        max_iters=max_iter,
        restarts=restarts,
        random_state=0,
        curve=True,
    ).fit(x, y)

def create_sa_nn(x, y, max_iter, temp):
    return NeuralNetwork(
        hidden_nodes=[x.shape[1], x.shape[1]],
        algorithm='simulated_annealing',
        max_iters=max_iter,
        schedule=GeomDecay(init_temp=temp),
        random_state=0,
        curve=True,
    ).fit(x, y)

def create_ga_nn(x, y, max_iter, mutation_prob):
    return NeuralNetwork(
        hidden_nodes=[x.shape[1], x.shape[1]],
        algorithm='genetic_alg',
        max_iters=max_iter,
        pop_size=x.shape[1],
        mutation_prob=mutation_prob,
        random_state=0,
        curve=True,
    ).fit(x, y)

def plot(x, name, label=None, legend_title=None, top_limit=None):
    plt = Plotter(
        name=name,
        learner='nn',
        axes={ 'x': 'Iterations', 'y': 'Fitness(x)' },
        legend_title=legend_title
    )
    # Training data
    plt.add_plot(
        x=np.arange(1, len(x)+1, dtype=int), 
        y=x, 
        label=label,
        marker=None
    )
    plt.save(top_limit=top_limit)

def sgd_learning_curve(episodes, x_train, y_train):
    lr = 0.0025
    sgd_losses = np.zeros(episodes)

    # Train NNs w/ sgd
    for i in tqdm(range(episodes)):
        sgd_losses[i] = create_sgd_nn(
            x=x_train,
            y=y_train,
            max_iter=i+1,
            lr=lr
        ).loss

    # Plot loss
    plot(
        x=sgd_losses,
        name='SGD - Loss vs Iterations',
        label=lr,
        legend_title='Learning rate'
    )

def sgd_fitness_curve(episodes, x_train, y_train, x_test):
    lr = 0.0025
    create_start = time.process_time()
    fitness_curve = create_sgd_nn(
        x=x_train,
        y=y_train,
        max_iter=episodes,
        lr=lr
    ).fitness_curve
    create_time = time.process_time() - create_start
    print('(SGD) time:', create_time)
    max_fit = max(fitness_curve)
    print('Best f(x) - SGD:', max_fit)
    print('Iter:', list(fitness_curve).index(max_fit)+1)

    plot(
        x=fitness_curve, 
        name='SGD - Fitness vs Iterations',
        label=lr,
        legend_title='Learning rate'
    )
    
def rhc_learning_curve(episodes, x_train, y_train):
    restarts = 0
    rhc_losses = np.zeros(episodes)

    # Train NNs w/ sgd
    for i in tqdm(range(episodes)):
        rhc_losses[i] = create_rhc_nn(
            x=x_train,
            y=y_train,
            max_iter=i+1,
            restarts=restarts
        ).loss

    # Plot loss
    plot(
        x=rhc_losses,
        name='RHC - Loss vs Iterations',
        label=restarts,
        legend_title='Restarts'
    )

def rhc_fitness_curve(episodes, x_train, y_train, x_test):
    restarts = 0
    create_start = time.process_time()
    fitness_curve = create_rhc_nn(
        x=x_train,
        y=y_train,
        max_iter=episodes,
        restarts=restarts
    ).fitness_curve
    create_time = time.process_time() - create_start
    print('(RHC) time:', create_time)
    max_fit = max(fitness_curve)
    print('Best f(x) - RHC:', max_fit)
    print('Iter:', list(fitness_curve).index(max_fit)+1)

    plot(
        x=fitness_curve, 
        name='RHC - Fitness vs Iterations',
        label=restarts,
        legend_title='Restarts'
    )

def sa_learning_curve(episodes, x_train, y_train):
    temp = 0.2
    sa_losses = np.zeros(episodes)

    # Train NNs w/ sgd
    for i in tqdm(range(episodes)):
        sa_losses[i] = create_sa_nn(
            x=x_train,
            y=y_train,
            max_iter=i+1,
            temp=temp
        ).loss

    # Plot loss
    plot(
        x=sa_losses,
        name='SA - Loss vs Iterations',
        label=temp,
        legend_title='Temperature'
    )

def sa_fitness_curve(episodes, x_train, y_train, x_test):
    temp = temp = 0.2
    create_start = time.process_time()
    fitness_curve = create_sa_nn(
        x=x_train,
        y=y_train,
        max_iter=episodes,
        temp=temp
    ).fitness_curve
    create_time = time.process_time() - create_start
    print('(SA) time:', create_time)
    max_fit = max(fitness_curve)
    print('Best f(x) - SA:', max_fit)
    print('Iter:', list(fitness_curve).index(max_fit)+1)

    plot(
        x=fitness_curve, 
        name='SA - Fitness vs Iterations',
        label=temp,
        legend_title='Temp'
    )

def ga_learning_curve(episodes, x_train, y_train):
    mutation_prob = 0.5
    ga_losses = np.zeros(episodes)

    # Train NNs w/ GA
    for i in tqdm(range(episodes)):
        ga_losses[i] = create_ga_nn(
            x=x_train,
            y=y_train,
            max_iter=i+1,
            mutation_prob=mutation_prob
        ).loss

    # Plot loss
    plot(
        x=ga_losses,
        name='GA - Loss vs Generations',
        label=mutation_prob,
        legend_title='Mutation prob'
    )

def ga_fitness_curve(episodes, x_train, y_train, x_test):
    mutation_prob = 0.5
    create_start = time.process_time()
    fitness_curve = create_ga_nn(
        x=x_train,
        y=y_train,
        max_iter=episodes,
        mutation_prob=mutation_prob
    ).fitness_curve
    create_time = time.process_time() - create_start
    print('(GA) time:', create_time)
    max_fit = max(fitness_curve)
    print('Best f(x) - GA:', max_fit)
    print('Iter:', list(fitness_curve).index(max_fit)+1)

    plot(
        x=fitness_curve, 
        name='GA - Fitness vs Generations',
        label=mutation_prob,
        legend_title='Mutation prob'
    )


if __name__ == "__main__":
    np.random.seed(93)

    # Ignore warnings
    warnings.warn = warn

    # Train-test split
    x_train, y_train, x_test, y_test = heart_failure_prediction()

    # Scale/encode data
    x_train, x_test = scale_data(x_train, x_test)
    y_train, y_test = encode_data(y_train, y_test)

    episodes = 500
   
    sgd_learning_curve(episodes, x_train, y_train)
    sgd_fitness_curve(500, x_train, y_train, x_test)

    rhc_learning_curve(episodes, x_train, y_train)
    rhc_fitness_curve(5000, x_train, y_train, x_test)

    sa_learning_curve(episodes, x_train, y_train)
    sa_fitness_curve(3000, x_train, y_train, x_test)

    ga_learning_curve(episodes, x_train, y_train)
    ga_fitness_curve(3000, x_train, y_train, x_test)