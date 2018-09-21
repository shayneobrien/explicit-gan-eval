import os, sys, json, itertools
import torch
import data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import partial
from collections import defaultdict
from models.model_utils import preprocess
from models.mnist_utils import preprocess_mnist

plt.switch_backend('agg')

"""
Results
"""

def get_multivariate_results(models, distributions, dimensions,
                             epochs, samples, hyperparameters):
    """ Multivariate distribution results """
    results, activation_type = nested_pickle_dict(), 'relu'
    for model_name, module in models.items():
        for dist in distributions:
            print('\n', model_name, dist, 'MULTIVARIATE')
            gen = data.Distribution(dist_type=dist, dim=dimensions)
            metrics = model_results(module, epochs, hyperparameters,
                                    gen, samples, dimensions, activation_type)
            results[model_name][dist].update(metrics)

    return results


def get_mixture_results(models, distributions, dimensions,
                        epochs, samples, n_mixtures, hyperparameters):
    """ Mixture model results """
    results, activation_type = nested_pickle_dict(), 'relu'
    for model_name, module in models.items():
        for dist_i in distributions[0:1]: # Just normal and other mixture models at the moment
            for dist_j in distributions:
                print('\n', model_name, dist_i, dist_j, "MIXTURE")
                gen = data.MixtureDistribution(dist_type=dist_i, mix_type=dist_j,
                                                n_mixtures=n_mixtures, dim=dimensions)

                metrics = model_results(module, epochs, hyperparameters,
                                        gen, samples, dimensions, activation_type)
                results[model_name][dist_i][dist_j].update(metrics)

    return results


def get_mnist_results(models, mnist_dim,
                      epochs, hyperparameters):
    """ Autoencoded MNIST results """

    # Unpack hyperparameters, initialize results dict
    lr, dim, bsize = hyperparameters
    results, activation_type = nested_pickle_dict(), 'sigmoid'

    # Create data iterators by training autoencoder on MNIST and using
    # its output as our 'ground truth'. We have to repredict for each
    # batch size, so remove the file if it exists already (force)
    if os.path.exists('data/autoencoder/cached_preds.txt'):
        os.remove('data/autoencoder/cached_preds.txt')
    train_iter, val_iter, test_iter = preprocess_mnist(BATCH_SIZE=bsize)

    # Normal passover routine
    for model_name, module in models.items():

        # Doesn't make sense to consider autoencoder vs. weaker autoencoder
        if model_name == 'autoencoder':
            continue
        print('\n', model_name, "MNIST")

        # Model, trainer, metrics
        model = module.Model(image_size=mnist_dim, hidden_dim=dim,
                             z_dim=int(round(max(dim/4, 1))), atype=activation_type)
        trainer = module.Trainer(model, train_iter, val_iter, test_iter)
        metrics = trainer.train(num_epochs=epochs, lr=lr)

        # Update metrics
        results[model_name]["mnist"].update(metrics)

    return results


def get_circle_results(models, dimensions,
                       epochs, samples, modes, n_circles, hyperparameters):
    """ Circles with random colors, sizes, locations results """

    # Unpack hyperparameters, initialize results dict
    lr, dim, bsize = hyperparameters
    results, activation_type = nested_pickle_dict(), 'sigmoid'

    for model_name, module in models.items():
        for n_circle in n_circles:
            for mode in modes:
                print('\n', model_name, n_circle, 'circles', mode, 'modes', 'CIRCLES')
                gen = data.CirclesDatasetGenerator(size=28, n_circles=n_circle, modes=mode,
                                                random_colors=False, random_sizes=False)

                metrics = model_results(module, epochs, hyperparameters,
                                        gen, samples, dimensions, activation_type)

                results[model_name][n_circle][mode].update(metrics)

    return results


def model_results(module, epochs, hyperparameters, gen, samples, dimensions, activation_type):
    """ Train a model, get metrics dictionary out """

    # Unpack hyperparameters
    lr, dim, bsize = hyperparameters

    # Create data iterators
    train_iter, val_iter, test_iter = preprocess(gen, samples, bsize)

    # Init model
    model = module.Model(image_size=dimensions,
                         hidden_dim=dim,
                         z_dim=int(round(max(dim/4, 1))),
                         atype=activation_type)

    # Init trainer
    trainer = module.Trainer(model,
                             train_iter,
                             val_iter,
                             test_iter)

    # Train and get output metrics
    metrics = trainer.train(num_epochs=epochs,
                            lr=lr)

    return metrics


"""
Best results
"""

def get_best_performance_multivariate(data_type, start_time, trial):
    """ For a trial, get the best performance for multivariate data """
    # Get path, files in path
    mypath = "hypertuning/{0}/{1}/trial_{2}".format(data_type, start_time, trial)
    files = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
    results = []

    # Read in the files
    for file in files:
        with open("{}/{}".format(mypath, file)) as f:
            data = json.load(f)
        results.append(data)

    # Initialize best dictionary
    optimal = nested_pickle_dict()

    # Go through all models, distributionss, metrics, and record the best
    for result in results:
        for model, distributions in result.items():
            for distribution, metrics in distributions.items():
                for metric, values in metrics.items():
                    if metric not in ["LR", "HDIM", "BSIZE"]:

                        # If metric is seen for the first time, it is the best
                        if metric not in optimal[model][distribution]:
                            optimal[model][distribution][metric]["value"] = values
                            optimal[model][distribution][metric]["parameters"] = [metrics["LR"], metrics["HDIM"], metrics["BSIZE"]]

                        # Otherwise, compare it the presently considered value
                        elif optimal[model][distribution][metric]["value"][-1] > values[-1]:
                            optimal[model][distribution][metric]["value"] = values
                            optimal[model][distribution][metric]["parameters"] = [metrics["LR"], metrics["HDIM"], metrics["BSIZE"]]

    return optimal


def get_best_performance_mixture(data_type, start_time, trial):
    """ For a trial, get the best performance for a mixture model """
    # Get path, files in path
    mypath = "hypertuning/{0}/{1}/trial_{2}".format(data_type, start_time, trial)
    files = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
    results = []

    # Read in the files
    for file in files:
        with open("{}/{}".format(mypath, file)) as f:
            data = json.load(f)
        results.append(data)

    # Initialize best dictionary
    optimal = nested_pickle_dict()
    for result in results:
        for model, mixtures in result.items():
            for mixture, distributions in mixtures.items():
                for distribution, metrics in distributions.items():
                    for metric, values in metrics.items():
                        if metric not in ["LR", "HDIM", "BSIZE"]:

                            # If metric is seen for the first time, it is the best
                            if metric not in optimal[model][mixture][distribution]:
                                optimal[model][mixture][distribution][metric]["value"] = values
                                optimal[model][mixture][distribution][metric]["parameters"] = [metrics["LR"], metrics["HDIM"], metrics["BSIZE"]]

                            # Otherwise, compare it the presently considered value
                            elif optimal[model][mixture][distribution][metric]["value"][-1] > values[-1]:
                                optimal[model][mixture][distribution][metric]["value"] = values
                                optimal[model][mixture][distribution][metric]["parameters"] = [metrics["LR"], metrics["HDIM"], metrics["BSIZE"]]

    return optimal


def get_best_performance_mnist(data_type, start_time, trial):
    """ For a trial, get the best performance for MNIST """
    return get_best_performance_multivariate(data_type, start_time, trial)


def get_best_performance_circles(data_type, start_time, trial):
    """ For a trial, get the best performance for circles """
    return get_best_performance_mixture(data_type, start_time, trial)


"""
Confidence intervals
"""

def get_confidence_intervals_multivariate(data_type, start_time):
    """ Compute 95% confidence intervals for multivariate """

    # Get file path and files therein
    mypath = "best/{0}/{1}/".format(data_type, start_time)
    files = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
    results = []

    # Load files
    for file in files:
        with open("{}/{}".format(mypath, file)) as f:
            data = json.load(f)
        results.append(data)

    # Get values for best runs
    optimal = nested_pickle_dict()
    for result in results:
        for model, distributions in result.items():
            for dist, metrics in distributions.items():
                for divergence, output in metrics.items():
                    if divergence not in optimal[model][dist]:
                        optimal[model][dist][divergence] = {'values': []}
                    optimal[model][dist][divergence]['values'].append(output['value'])

    # Compute 5th and 95th percentiles for each epoch
    for result in results:
        for model, distributions in result.items():
            for dist, metrics in distributions.items():
                for divergence, output in metrics.items():
                    data = np.array(optimal[model][dist][divergence]['values'])
                    optimal[model][dist][divergence]['5'] = list(np.percentile(data, 5, axis=0))
                    optimal[model][dist][divergence]['95'] = list(np.percentile(data, 95, axis=0))

    return optimal


def get_confidence_intervals_mixture(data_type, start_time):
    """ Compute 95% confidence intervals for mixtures """

    # Get file path and files therein
    mypath = "best/{0}/{1}/".format(data_type, start_time)
    files = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
    results = []

    # Load files
    for file in files:
        with open("{}/{}".format(mypath, file)) as f:
            data = json.load(f)
        results.append(data)

    # Get values for best runs
    optimal = nested_pickle_dict()
    for result in results:
        for model, distributions in result.items():
            for dist_i, dists_j in distributions.items():
                for dist_j, metrics in dists_j.items():
                    for divergence, output in metrics.items():
                        if divergence not in optimal[model][dist_i][dist_j]:
                            optimal[model][dist_i][dist_j][divergence] = {'values': []}
                        optimal[model][dist_i][dist_j][divergence]['values'].append(output['value'])

    # Compute 5th and 95th percentiles for each epoch
    for result in results:
        for model, distributions in result.items():
            for dist_i, dists_j in distributions.items():
                for dist_j, metrics in dists_j.items():
                    for divergence, output in metrics.items():
                        data = np.array(optimal[model][dist_i][dist_j][divergence]['values'])
                        optimal[model][dist_i][dist_j][divergence]['5'] = list(np.percentile(data, 5, axis=0))
                        optimal[model][dist_i][dist_j][divergence]['95'] = list(np.percentile(data, 95, axis=0))

    return optimal


def get_confidence_intervals_mnist(data_type):
    """ Compute 95% confidence intervals for MNIST """
    return get_confidence_intervals_multivariate(data_type)


def get_confidence_intervals_circles(data_type):
    """ Compute 95% confidence intervals for circles """
    return get_confidence_intervals_mixture(data_type)


"""
Best results graphs
"""
#TODO: Fix all of these

def get_best_graph(results,
                   models,
                   distributions,
                   distance_metrics,
                   num_epochs):
    # TODO: fix save error, legend, make pretty
    for metric in distance_metrics:
        for model_name, module in models.items():
            for dist in distributions:
                data = results[model_name][dist][metric]['value']
                print(model_name, dist, metric, data)
                plt.plot(np.linspace(1, num_epochs, len(data)), data, label=dist)

            plt.xlabel("Epoch")
            plt.ylabel(metric)
            plt.title("{0}: {1}".format(model_name.upper(), metric))
            plt.legend(loc="best")
            plt.savefig('graphs/multivariate/{0}_{1}.png'.format(metric, model_name), dpi=100)
            plt.clf()

def get_multivariate_graphs(results, models, distributions,
                            distance_metrics, num_epochs):
    # TODO: fix save error, legend, make pretty
    for model_name, module in models.items():
        for dist in distributions:
            for metric in distance_metrics:
                data = results[model_name][dist][metric]
                print(model_name, dist, metric, data)
                plt.plot(np.linspace(1, num_epochs, len(data)), data)

            plt.xlabel("Epoch")
            plt.ylabel(metric)
            plt.title("{0}: {1}".format(model_name.upper(), metric))
            plt.legend()
            plt.savefig('graphs/multivariate/{0}_{1}.png'.format(model_name, metric), dpi=100)
            plt.clf()


def get_mixture_graphs(results, models, distributions,
                        distance_metrics, num_epochs):
    # TODO: fix save error, legend, make pretty
    for model_name, module in models.items():
        for dist_i in distributions[0:1]: # Just normal and other mixture models at the moment
            for dist_j in distributions:
                for metric in distance_metrics:
                    data = results[model_name][dist_i][dist_j][metric]
                    plt.plot(np.linspace(1, num_epochs, len(data)), data)

    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.title("{0}: {1}-{2}".format(model_name.upper(), dist_i, dist_j))
    plt.legend()
    plt.savefig('graphs/mixture/{0}_{1}-{2}.png'.format(model_name, dist_i, dist_j), dpi=100)
    plt.clf()


def get_mnist_graphs(results, models, distance_metrics, num_epochs):
    for model_name, module in models.items():
        normal = pd.DataFrame(results[model_name]['mnist'])
        for dist in distance_metrics:
            plt.plot(range(len(normal['mnist'])), normal['mnist'], label="MNIST")

        plt.xlabel("Epoch")
        plt.ylabel(dist)
        plt.title("{0}: {1}".format(model_name.upper(), dist))
        plt.legend()
        plt.savefig('graphs/mnist/{0}_{1}.png'.format(model_name, dist), dpi=100)
        plt.clf()

def get_circles_graph(results, models, ):
    pass


"""
Misc. utility function
"""

def nested_pickle_dict():
    """ Picklable defaultdict nested dictionaries """
    return defaultdict(nested_pickle_dict)
