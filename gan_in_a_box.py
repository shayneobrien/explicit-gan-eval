import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset
from scipy import stats
from scipy.stats import entropy, ks_2samp, moment, wasserstein_distance, energy_distance
matplotlib.use('agg')
import matplotlib.pyplot as plt

import data

import gans.w_gan as wgan
import gans.w_gp_gan as wgpgan
import gans.vae as vae
import gans.ns_gan as nsgan
import gans.mm_gan as mmgan
import gans.ls_gan as lsgan
import gans.dra_gan as dragan
import gans.be_gan as began
from gans.load_data import get_data
from gans.utils import to_var, get_pdf, get_the_data, preprocess
from gans.load_data import get_data
import json
import itertools


def get_multivariate_results(gans, gans_index, distributions, dimensions, epochs, samples, hyperparameters):
    res = {}
    lr, dim, step, bsize = hyperparameters
    for index, gan in enumerate(gans[:-1]):
        res[gans_index[index]] = {}
        print(gans_index[index])
        for dist in distributions:
            res[gans_index[index]][dist] = {}
            gen = data.Distribution(dist, dimensions)
            # train_iter, val_iter, test_iter = preprocess(gen, samples, bsize)
            train_iter, val_iter, test_iter = preprocess(gen, samples, 100)  # Default batch size
            if gans_index[index] == "vae":
                model = vae.VAE(image_size=dimensions, hidden_dim=dim, z_dim=20)
                if torch.cuda.is_available():
                    model.cuda()
                trainer = vae.Trainer(train_iter, val_iter, test_iter)
                model, kl, ks, js, wd, ed = trainer.train(model, num_epochs=epochs)
            else:
                model = gan.GAN(image_size=dimensions, hidden_dim=dim, z_dim=int(round(dimensions/4, 0)))
                if torch.cuda.is_available():
                    model = model.cuda()
                trainer = gan.Trainer(train_iter, val_iter, test_iter)
                # model, kl, ks, js, wd, ed = trainer.train(model=model, num_epochs=epochs, G_lr=lr, D_lr=lr, D_steps=step)
                model, kl, ks, js, wd, ed = trainer.train(model=model, num_epochs=epochs)  ## Default lr and step size
            res[gans_index[index]][dist]["KL-Divergence"] = kl
            res[gans_index[index]][dist]["Jensen-Shannon"] = js
            res[gans_index[index]][dist]["Wasserstein-Distance"] = wd
            res[gans_index[index]][dist]["Energy-Distance"] = ed
            # Hyperparams
            # res[gans_index[index]][dist]["LR"] = lr
            # res[gans_index[index]][dist]["HDIM"] = dim
            # res[gans_index[index]][dist]["DSTEP"] = step
            # res[gans_index[index]][dist]["BSIZE"] = bsize
    return res


def get_mixture_results(gans, gans_index, distributions, dimensions, epochs, samples, n_mixtures):
    res = {}
    for index, gan in enumerate(gans[:-1]):
        res[gans_index[index]] = {}
        print(gans_index[index])
        for dist_i in distributions[0]:  # Just normal and other mixture models at the moment
            res[gans_index[index]][dist_i] = {}
            for dist_j in distributions:  
                print(dist_j)
                res[gans_index[index]][dist_i][dist_j] = {}
                print(dist_i, dist_j, n_mixtures, dimensions, samples)
                gen = data.MixtureDistribution(dist_i, dist_j, n_mixtures=n_mixtures, dim=dimensions)
                train_iter, val_iter, test_iter = preprocess(gen, samples) # some error is occuring here
                if gans_index[index] == "vae":
                    model = vae.VAE(image_size=dimensions, hidden_dim=400, z_dim=20)
                    if torch.cuda.is_available():
                        model.cuda()
                    trainer = vae.Trainer(train_iter, val_iter, test_iter)
                    model, kl, ks, js, wd, ed = trainer.train(model, num_epochs=epochs)
                else:
                    model = gan.GAN(image_size=dimensions, hidden_dim=256, z_dim=int(round(dimensions/4, 0)))
                    if torch.cuda.is_available():
                        model = model.cuda()
                    trainer = gan.Trainer(train_iter, val_iter, test_iter)
                    model, kl, ks, js, wd, ed = trainer.train(model=model, num_epochs=epochs)
                res[gans_index[index]][dist_i][dist_j]["KL-Divergence"] = kl
                res[gans_index[index]][dist_i][dist_j]["Jensen-Shannon"] = js
                res[gans_index[index]][dist_i][dist_j]["Wasserstein-Distance"] = wd
                res[gans_index[index]][dist_i][dist_j]["Energy-Distance"] = ed
    return res


def get_circle_results(gans, gans_index, dimensions, epochs, samples):
    res = {}
    for index, gan in enumerate(gans[:-1]):
        res[gans_index[index]] = {}
        print(gans_index[index])
        res[gans_index[index]]["circle"] = {}
        generator = data.CirclesDatasetGenerator(size=dimensions, n_circles=samples, random_colors=True, random_sizes=True, modes=20)
        train_iter, val_iter, test_iter = preprocess(generator, samples)
        if gans_index[index] == "vae":
            model = vae.VAE(image_size=dimensions, hidden_dim=400, z_dim=20)
            if torch.cuda.is_available():
                model.cuda()
            trainer = vae.Trainer(train_iter, val_iter, test_iter)
            model, kl, ks, js, wd, ed = trainer.train(model, num_epochs=epochs)
        else:
            model = gan.GAN(image_size=dimensions, hidden_dim=256, z_dim=int(round(dimensions/4, 0)))
            if torch.cuda.is_available():
                model = model.cuda()
            trainer = gan.Trainer(train_iter, val_iter, test_iter)
            model, kl, ks, js, wd, ed = trainer.train(model=model, num_epochs=epochs)
        res[gans_index[index]]["circle"]["KL-Divergence"] = kl
        res[gans_index[index]]["circle"]["Jensen-Shannon"] = js
        res[gans_index[index]]["circle"]["Wasserstein-Distance"] = wd
        res[gans_index[index]]["circle"]["Energy-Distance"] = ed
    return res


def get_mnist_results(gans, gans_index, epochs):
    res = {}
    for index, gan in enumerate(gans[:-1]):
        res[gans_index[index]] = {}
        print(gans_index[index])
        res[gans_index[index]]["mnist"] = {}
        train_iter, val_iter, test_iter = get_data(100)
        if gans_index[index] == "vae":
            model = vae.VAE(image_size=784, hidden_dim=400, z_dim=20)
            if torch.cuda.is_available():
                model.cuda()
            trainer = vae.Trainer(train_iter, val_iter, test_iter)
            model, kl, ks, js, wd, ed = trainer.train(model, num_epochs=epochs)
        else:
            model = gan.GAN(image_size=784, hidden_dim=256, z_dim=int(round(dimensions/4, 0)))
            if torch.cuda.is_available():
                model = model.cuda()
            trainer = gan.Trainer(train_iter, val_iter, test_iter)
            model, kl, ks, js, wd, ed = trainer.train(model=model, num_epochs=epochs)
        res[gans_index[index]]["mnist"]["KL-Divergence"] = kl
        res[gans_index[index]]["mnist"]["Jensen-Shannon"] = js
        res[gans_index[index]]["mnist"]["Wasserstein-Distance"] = wd
        res[gans_index[index]]["mnist"]["Energy-Distance"] = ed
    return res


def get_multivariate_graphs(res, gans_index, distance_metrics):
    for gan in gans_index[:-1]:
        normal = pd.DataFrame(res[gan]['normal'])
        beta = pd.DataFrame(res[gan]['beta'])
        exponential = pd.DataFrame(res[gan]['exponential'])
        gamma = pd.DataFrame(res[gan]['gamma'])
        gumbel = pd.DataFrame(res[gan]['gumbel'])
        for dist in distance_metrics:
            print(range(len(normal[dist]))
            plt.plot(range(len(normal[dist])), normal[dist], label="Normal")
            plt.plot(range(len(normal[dist])), beta[dist], label="Beta")
            plt.plot(range(len(normal[dist])), exponential[dist], label="Exponential")
            plt.plot(range(len(normal[dist])), gamma[dist], label="Gamma")
            plt.plot(range(len(normal[dist])), gumbel[dist], label="Gumbel")
            plt.xlabel("Epoch")
            plt.ylabel(dist)
            plt.title("{0}: {1}".format(gan.upper(), dist))
            plt.legend()
            plt.savefig('graphs/{0}_{1}.png'.format(gan, dist), dpi=100)
            plt.clf()


def get_mixture_graphs(res, gans_index, distance_metrics):
    for gan in gans_index[:-1]:
        normal_normal = pd.DataFrame(res[gan]['normal']['normal'])
        normal_beta = pd.DataFrame(res[gan]['normal']['beta'])
        normal_exponential = pd.DataFrame(res[gan]['normal']['exponential'])
        normal_gamma = pd.DataFrame(res[gan]['normal']['gamma'])
        normal_gumbel = pd.DataFrame(res[gan]['normal']['gumbel'])
        for dist in distance_metrics:
            plt.plot(range(epochs), normal_normal[dist], label="Normal")
            plt.plot(range(epochs), normal_beta[dist], label="Beta")
            plt.plot(range(epochs), normal_exponential[dist], label="Exponential")
            plt.plot(range(epochs), normal_gamma[dist], label="Gamma")
            plt.plot(range(epochs), normal_gumbel[dist], label="Gumbel")
            plt.xlabel("Epoch")
            plt.ylabel(dist)
            plt.title("{0}: {1}".format(gan.upper(), dist))
            plt.legend()
            plt.savefig('graphs/{0}_{1}.png'.format(gan, dist), dpi=100)
            plt.clf()


def get_mnist_graphs(res, gans_index, distance_metrics):
    for gan in gans_index[:-1]:
        normal = pd.DataFrame(res[gan]['mnist'])
        for dist in distance_metrics:
            plt.plot(range(len(normal['mnist'])), normal['mnist'], label="MNIST")
            plt.xlabel("Epoch")
            plt.ylabel(dist)
            plt.title("{0}: {1}".format(gan.upper(), dist))
            plt.legend()
            plt.savefig('graphs/{0}_{1}.png'.format(gan, dist), dpi=100)
            plt.clf()


if __name__ == "__main__":

    hyperparam = [5e-3, 256, 5, 10]
    learning_rates = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
    hidden_dims = [16, 32, 64, 128, 256]
    D_steps = [1, 2, 3, 4, 5]
    BATCH_SIZE = [10, 25, 50, 75, 100, 150, 200, 250]

    print("Choose a dataset: multivariate, mixture, or circles")
    print("e.g. python gan_in_a_box.py multivariate n_dimensions n_epochs n_samples")
    data_type = sys.argv[1]
    dimensions = int(sys.argv[2])
    epochs = int(sys.argv[3])
    samples = int(sys.argv[4])
    if data_type == "mixture":
        print("adding n_mixtures")
        n_mixtures = int(sys.argv[5])
    print("python gan_in_a_box.py {0} {1} {2}".format(dimensions, epochs, samples))
    distributions = ['normal', 'beta', 'exponential', 'gamma', 'gumbel', 'laplace']
    gans = [wgan, wgpgan, nsgan, lsgan, mmgan, began, vae]
    gans_index = ["wgan",  "wgpgan", "nsgan", "lsgan", "mmgan", "began", "vae"]
    distance_metrics = ["KL-Divergence", "Jensen-Shannon", "Wasserstein-Distance", "Energy-Distance"]
    if data_type == "multivariate":
        # for hyperparam in list(itertools.product(*[learning_rates, hidden_dims, D_steps, BATCH_SIZE])):
        #     lr, dim, step, bsize = hyperparam
        #     print(hyperparam)
        res = get_multivariate_results(gans, gans_index, distributions, dimensions, epochs, samples, hyperparam)
        print(type(res))
        with open('hypertuning/data{0}.json'.format(str(hyperparam)), 'w') as outfile:
            json.dump(res, outfile)
        get_multivariate_graphs(res, gans_index, distance_metrics)
    elif data_type == "mixture":
        res = get_mixture_results(gans, gans_index, distributions, dimensions, epochs, samples, n_mixtures)
        get_mixture_graphs(res, gans_index, distance_metrics)
    elif data_type == "circles":
        res = get_circle_results(gans, gans_index, dimensions, epochs, samples)
        print("to do: graph circles")
    elif data_type == "mnist":
        res = get_mnist_results(gans, gans_index, epochs)
        get_mnist_graphs(res, gans_index, distance_metrics)
        print("to do: graph circles")
    print("Le Fin.")
