import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset
from scipy import stats
from scipy.stats import entropy, ks_2samp, moment, wasserstein_distance, energy_distance
import matplotlib.pyplot as plt
plt.switch_backend('agg')

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
    lr, dim, bsize = hyperparameters
    for index, gan in enumerate(gans[:-1]):
        res[gans_index[index]] = {}
        print(gans_index[index])
        for dist in distributions:
            res[gans_index[index]][dist] = {}
            gen = data.Distribution(dist, dimensions)
            train_iter, val_iter, test_iter = preprocess(gen, samples, bsize)
            # train_iter, val_iter, test_iter = preprocess(gen, samples, 100)  # Default batch size
            if gans_index[index] == "vae":
                model = vae.VAE(image_size=dimensions, hidden_dim=dim, z_dim=20)
                if torch.cuda.is_available():
                    model.cuda()
                trainer = vae.Trainer(train_iter, val_iter, test_iter)
                model, kl, ks, js, wd, ed, dl, gl = trainer.train(model, num_epochs=epochs)
            else:
                model = gan.GAN(image_size=dimensions, hidden_dim=dim, z_dim=int(round(dimensions/4, 0)))
                if torch.cuda.is_available():
                    model = model.cuda()
                trainer = gan.Trainer(train_iter, val_iter, test_iter)
                model, kl, ks, js, wd, ed, dl, gl = trainer.train(model=model, num_epochs=epochs, G_lr=lr, D_lr=lr)
                # model, kl, ks, js, wd, ed = trainer.train(model=model, num_epochs=epochs)  ## Default lr and step size
            res[gans_index[index]][dist]["KL-Divergence"] = kl
            res[gans_index[index]][dist]["Jensen-Shannon"] = js
            res[gans_index[index]][dist]["Wasserstein-Distance"] = wd
            res[gans_index[index]][dist]["Energy-Distance"] = ed
            res[gans_index[index]][dist]["DLoss"] = dl
            res[gans_index[index]][dist]["GLoss"] = gl
            # Hyperparams
            res[gans_index[index]][dist]["LR"] = lr
            res[gans_index[index]][dist]["HDIM"] = dim
            # res[gans_index[index]][dist]["DSTEP"] = step
            res[gans_index[index]][dist]["BSIZE"] = bsize
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
        print("\n\n\n")
        res[gans_index[index]]["mnist"] = {}
        train_iter, val_iter, test_iter = get_data(2000)
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
            trainer = gan.Trainer(train_iter, val_iter, test_iter, mnist=True)
            model, kl, ks, js, wd, ed, dl, gl = trainer.train(model=model, num_epochs=epochs)
        res[gans_index[index]]["mnist"]["KL-Divergence"] = kl
        res[gans_index[index]]["mnist"]["Jensen-Shannon"] = js
        res[gans_index[index]]["mnist"]["Wasserstein-Distance"] = wd
        res[gans_index[index]]["mnist"]["Energy-Distance"] = ed
        res[gans_index[index]]["mnist"]["DLoss"] = dl
        res[gans_index[index]]["mnist"]["GLoss"] = gl
    return res


def get_multivariate_graphs(res, gans_index, distance_metrics):
    for gan in gans_index[:-1]:
        normal = pd.DataFrame(res[gan]['normal'])
        beta = pd.DataFrame(res[gan]['beta'])
        exponential = pd.DataFrame(res[gan]['exponential'])
        gamma = pd.DataFrame(res[gan]['gamma'])
        gumbel = pd.DataFrame(res[gan]['gumbel'])
        for dist in distance_metrics:
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
