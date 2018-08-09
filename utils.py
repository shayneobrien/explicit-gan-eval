import os, sys, json, itertools
import torch
import data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import partial
from collections import defaultdict
from models.gan_utils import preprocess
from models.gan_utils_mnist import preprocess_mnist

plt.switch_backend('agg')


def get_multivariate_results(models, distributions, dimensions,
                             epochs, samples, hyperparameters):
    results = nested_pickle_dict()
    for model_name, module in models.items():
        for dist in distributions:
            print('\n', model_name, dist)
            gen = data.Distribution(dist_type=dist, dim=dimensions)
            metrics = model_results(module, epochs, hyperparameters,
                                    gen, samples, dimensions)
            results[model_name][dist].update(metrics)
    return results


def get_mixture_results(models, distributions, dimensions,
                        epochs, samples, n_mixtures, hyperparameters):
    results = nested_pickle_dict()
    for model_name, module in models.items():
        for dist_i in distributions[0:1]: # Just normal and other mixture models at the moment
            for dist_j in distributions:
                # TODO: Fix mix_type='uniform', or mix_type = 'random'
                gen = data.MixtureDistribution(dist_type=dist_i, mix_type=dist_j,
                                                n_mixtures=n_mixtures, dim=dimensions)

                metrics = model_results(module, epochs, hyperparameters,
                                        gen, samples, dimensions)
                results[model_name][dist_i][dist_j].update(metrics)

    return results


def get_circle_results(models, dimensions,
                       epochs, samples, hyperparameters):
    results = nested_pickle_dict()
    for model_name, module in models.items():
        gen = data.CirclesDatasetGenerator(size=dimensions, n_circles=samples,
                                            random_colors=True, random_sizes=True,
                                            modes=20)
        metrics = model_results(module, epochs, hyperparameters,
                                gen, samples, dimensions)
        results[model_name].update(metrics)

    return results


def model_results(module, epochs, hyperparameters, gen, samples, dimensions):
    """ Train a model, get metrics dictionary out """
    # Unpack hyperparameters, initialize results dictionary
    lr, dim, bsize = hyperparameters

    # Create data iterators
    train_iter, val_iter, test_iter = preprocess(gen, samples, bsize)

    # Model, trainer, metrics
    model = module.Model(image_size=dimensions, hidden_dim=dim, z_dim=int(round(dimensions/4, 0)))
    trainer = module.Trainer(model, train_iter, val_iter, test_iter)
    metrics = trainer.train(num_epochs=epochs, lr=lr)

    return metrics


def get_best_performance(data_type):
    mypath = "hypertuning/{}".format(data_type)
    files = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
    results = []
    for file in files:
        with open("{}/{}".format(mypath, file)) as f:
            data = json.load(f)
        results.append(data)
    optimal = nested_pickle_dict()
    for result in results:
        for gan, distributions in result.items():
            for distribution, metrics in distributions.items():
                for metric, values in metrics.items():
                    if metric not in ["LR", "HDIM", "BSIZE"]:
                        if metric not in optimal[gan][distribution]:
                            optimal[gan][distribution][metric]["value"] = values
                            optimal[gan][distribution][metric]["parameters"] = [metrics["LR"], metrics["HDIM"], metrics["BSIZE"]]
                        elif optimal[gan][distribution][metric]["value"][-1] > values[-1]:
                            optimal[gan][distribution][metric]["value"] = values
                            optimal[gan][distribution][metric]["parameters"] = [metrics["LR"], metrics["HDIM"], metrics["BSIZE"]]

    return optimal


def get_confidence_intervals(data_type):
    mypath = "best/{}".format(data_type)
    files = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
    results = []
    for file in files:
        with open("{}/{}".format(mypath, file)) as f:
            data = json.load(f)
        results.append(data)
    optimal = {}
    for result in results:
        for gan, distributions in result.items():
            if gan not in optimal:
                optimal[gan] = {}
            for distribution, metrics in distributions.items():
                if distribution not in optimal[gan]:
                    optimal[gan][distribution] = {}
                for metric, values in metrics.items():
                    if metric not in optimal[gan][distribution]:
                        optimal[gan][distribution][metric] = {"original": []}
                    optimal[gan][distribution][metric]["original"].append(values['value'])
    for result in results:
        for gan, distributions in result.items():
            for distribution, metrics in distributions.items():
                for metric, values in metrics.items():
                    data = np.array(optimal[gan][distribution][metric]["original"])
                    optimal[gan][distribution][metric]['5'] = list(np.percentile(data, 5, axis=0))
                    optimal[gan][distribution][metric]['95'] = list(np.percentile(data, 95, axis=0))
    return optimal

# def get_circle_results(gans, dimensions, epochs, samples):
#     res = {}
#     for key, gan in gans.items():
#         res[key] = {}
#         print(key)
#         res[key]["circle"] = {}
#         generator = data.CirclesDatasetGenerator(size=dimensions, n_circles=samples, random_colors=True, random_sizes=True, modes=20)
#         train_iter, val_iter, test_iter = preprocess(generator, samples)
#         if key == "vae":
#             continue
#             # model = vae.VAE(image_size=dimensions, hidden_dim=400, z_dim=20)
#             # trainer = vae.Trainer(model, train_iter, val_iter, test_iter)
#             # model, kl, ks, js, wd, ed = trainer.train(model, num_epochs=epochs)
#         else:
#             model = gan.GAN(image_size=dimensions, hidden_dim=256, z_dim=int(round(dimensions/4, 0)))
#             trainer = gan.Trainer(model, train_iter, val_iter, test_iter)
#             model, kl, ks, js, wd, ed = trainer.train(model=model, num_epochs=epochs)
#         res[key]["circle"]["KL-Divergence"] = kl
#         res[key]["circle"]["Jensen-Shannon"] = js
#         res[key]["circle"]["Wasserstein-Distance"] = wd
#         res[key]["circle"]["Energy-Distance"] = ed
#     return res
#
#
def model_results_mnist(module, epochs, hyperparameters, dimensions):
    """ Train a model, get metrics dictionary out """
    # Unpack hyperparameters, initialize results dictionary
    lr, dim, bsize = hyperparameters

    # Create data iterators
    train_iter, val_iter, test_iter = preprocess_mnist(bsize)

    # Model, trainer, metrics
    model = module.Model(image_size=dimensions,
                        hidden_dim=dim,
                        z_dim=int(round(dimensions/4)))

    trainer = module.Trainer(model,
                            train_iter,
                            val_iter,
                            test_iter)

    metrics = trainer.train(num_epochs=epochs,
                            lr=lr)  # ... this: ValueError: too many values to unpack (expected 2)

    return metrics


def get_mnist_results(models, dimensions,
                             epochs, hyperparameters):
    results = nested_pickle_dict()
    for model_name, module in models.items():
        print('\n', model_name, "mnist")
        metrics = model_results_mnist(module, epochs, hyperparameters, dimensions)
        results[model_name]["mnist"].update(metrics)
    return results


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
    plt.title("{0}: {1}".format(model_name.upper(), dist))
    plt.legend()
    plt.savefig('graphs/mutlivariate/{0}_{1}.png'.format(model_name, dist), dpi=100)
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


# def get_mnist_graphs(res, gans_index, distance_metrics):
#     for gan, value in gans.items():
#         normal = pd.DataFrame(res[gan]['mnist'])
#         for dist in distance_metrics:
#             plt.plot(range(len(normal['mnist'])), normal['mnist'], label="MNIST")
#             plt.xlabel("Epoch")
#             plt.ylabel(dist)
#             plt.title("{0}: {1}".format(gan.upper(), dist))
#             plt.legend()
#             plt.savefig('graphs/{0}_{1}.png'.format(gan, dist), dpi=100)
#             plt.clf()


def nested_pickle_dict():
    """ Picklable defaultdict nested dictionaries """
    return defaultdict(nested_pickle_dict)
