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
        mutation_prob=mutation_prob,
        random_state=0,
        curve=True,
    ).fit(x, y)

def plot(data, name, labels, legend_title=None, top_limit=None):
    plt = Plotter(
        name=name,
        learner='nn',
        axes={ 'x': 'Iterations', 'y': 'Loss' },
        legend_title=legend_title
    )
    for i in range(len(data)):
        plt.add_plot(
            x=np.arange(1, len(data[i])+1, dtype=int), 
            y=data[i], 
            label=labels[i],
            marker=None
        )
    plt.save(top_limit=top_limit)

def sgd_learning_curve(episodes, x_train, y_train):
    lrs = [0.0025]
    sgd_losses = np.zeros((len(lrs), episodes))

    # Train NNs w/ sgd
    for i in tqdm(range(episodes)):
        for lr in range(len(lrs)):
            sgd_losses[lr][i] = create_sgd_nn(
                x=x_train,
                y=y_train,
                max_iter=i+1,
                lr=lrs[lr]
            ).loss

    # Plot loss
    plot(
        data=sgd_losses,
        name='Learning Curve - SGD',
        labels=[lr for lr in lrs],
        legend_title='Learning rate',
        # top_limit=2.
    )

    print('SGD - Min loss (train):', min(sgd_losses.flatten()))

def rhc_learning_curve(episodes, x_train, y_train):
    restarts = [0]
    rhc_losses = np.zeros((len(restarts), episodes))

    # Train NNs w/ sgd
    for i in tqdm(range(episodes)):
        for r in range(len(restarts)):
            rhc_losses[r][i] = create_rhc_nn(
                x=x_train,
                y=y_train,
                max_iter=i+1,
                restarts=restarts[r]
            ).loss

    # Plot loss
    plot(
        data=rhc_losses,
        name='Learning Curve - RHC',
        labels=['Randomized Hill Climbing']
    )

    print('RHC - Min loss (train):', min(rhc_losses.flatten()))

def sa_learning_curve(episodes, x_train, y_train):
    temps = [0.2]
    sa_losses = np.zeros((len(temps), episodes))

    # Train NNs w/ sgd
    for i in tqdm(range(episodes)):
        for t in range(len(temps)):
            sa_losses[t][i] = create_sa_nn(
                x=x_train,
                y=y_train,
                max_iter=i+1,
                temp=temps[t]
            ).loss

    # Plot loss
    plot(
        data=sa_losses,
        name='Learning Curve - SA',
        labels=[t for t in temps],
        legend_title='Temperature'
    )

    print('SA - Min loss (train):', min(sa_losses.flatten()))

def ga_learning_curve(episodes, x_train, y_train):
    mutation_probs = [0.5]
    ga_losses = np.zeros((len(mutation_probs), episodes))

    # Train NNs w/ GA
    for i in tqdm(range(episodes)):
        for t in range(len(mutation_probs)):
            ga_losses[t][i] = create_ga_nn(
                x=x_train,
                y=y_train,
                max_iter=i+1,
                mutation_prob=mutation_probs[t]
            ).loss

    # Plot loss
    plot(
        data=ga_losses,
        name='Learning Curve - GA',
        labels=[mp for mp in mutation_probs],
        legend_title='Mutation prob'
    )

    print('GA - Min loss (train):', min(ga_losses.flatten()))

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
   
    # sgd_learning_curve(episodes, x_train, y_train)
    # rhc_learning_curve(episodes, x_train, y_train)
    # sa_learning_curve(episodes, x_train, y_train)
    ga_learning_curve(episodes, x_train, y_train)