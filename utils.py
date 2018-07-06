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


def get_multivariate_results(gans, distributions, dimensions, epochs, samples, hyperparameters):
    res = {}
    lr, dim, bsize = hyperparameters
    for key, gan in gans.items():
        res[key] = {}
        print(key)
        for dist in distributions:
            res[key][dist] = {}
            gen = data.Distribution(dist, dimensions)
            train_iter, val_iter, test_iter = preprocess(gen, samples, bsize)
            print(train_iter)
            print(val_iter)
            print(test_iter)
            if key == "vae":
                model = vae.VAE(image_size=dimensions, hidden_dim=dim, z_dim=20)
                if torch.cuda.is_available():
                    model.cuda()
                trainer = vae.Trainer(train_iter, val_iter, test_iter)
                model, kl, ks, js, wd, ed, dl, gl = trainer.train(model, num_epochs=epochs)
            else:
                model = gan.GAN(image_size=dimensions, hidden_dim=dim, z_dim=int(round(dimensions/4, 0)))
                if torch.cuda.is_available():
                    model = model.cuda()
                trainer = gan.Trainer(model, train_iter, val_iter, test_iter)
                metrics = trainer.train(num_epochs=epochs, G_lr=lr, D_lr=lr)
            res[key][dist]["KL-Divergence"] = metrics['kl']
            res[key][dist]["Jensen-Shannon"] = metrics['js']
            res[key][dist]["Wasserstein-Distance"] = metrics['wd']
            res[key][dist]["Energy-Distance"] = metrics['ed']
            res[key][dist]["DLoss"] = metrics['dloss']
            res[key][dist]["GLoss"] = metrics['gloss']
            # Hyperparams
            res[key][dist]["LR"] = lr
            res[key][dist]["HDIM"] = dim
            res[key][dist]["BSIZE"] = bsize
    return res


def get_mixture_results(gans, distributions, dimensions, epochs, samples, n_mixtures):
    res = {}
    for key, gan in gans.items():
        res[key] = {}
        print(key)
        for dist_i in distributions[0]:  # Just normal and other mixture models at the moment
            res[key][dist_i] = {}
            for dist_j in distributions:  
                print(dist_j)
                res[key][dist_i][dist_j] = {}
                print(dist_i, dist_j, n_mixtures, dimensions, samples)
                gen = data.MixtureDistribution(dist_i, dist_j, n_mixtures=n_mixtures, dim=dimensions)
                train_iter, val_iter, test_iter = preprocess(gen, samples) # some error is occuring here
                if key == "vae":
                    model = vae.VAE(image_size=dimensions, hidden_dim=400, z_dim=20)
                    if torch.cuda.is_available():
                        model.cuda()
                    trainer = vae.Trainer(train_iter, val_iter, test_iter)
                    model, kl, ks, js, wd, ed = trainer.train(model, num_epochs=epochs)
                else:
                    model = value.GAN(image_size=dimensions, hidden_dim=256, z_dim=int(round(dimensions/4, 0)))
                    if torch.cuda.is_available():
                        model = model.cuda()
                    trainer = gan.Trainer(train_iter, val_iter, test_iter)
                    model, kl, ks, js, wd, ed = trainer.train(model=model, num_epochs=epochs)
                res[key][dist_i][dist_j]["KL-Divergence"] = kl
                res[key][dist_i][dist_j]["Jensen-Shannon"] = js
                res[key][dist_i][dist_j]["Wasserstein-Distance"] = wd
                res[key][dist_i][dist_j]["Energy-Distance"] = ed
    return res


def get_circle_results(gans, dimensions, epochs, samples):
    res = {}
    for key, gan in gans.items():
        res[key] = {}
        print(key)
        res[key]["circle"] = {}
        generator = data.CirclesDatasetGenerator(size=dimensions, n_circles=samples, random_colors=True, random_sizes=True, modes=20)
        train_iter, val_iter, test_iter = preprocess(generator, samples)
        if key == "vae":
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
        res[key]["circle"]["KL-Divergence"] = kl
        res[key]["circle"]["Jensen-Shannon"] = js
        res[key]["circle"]["Wasserstein-Distance"] = wd
        res[key]["circle"]["Energy-Distance"] = ed
    return res


def get_mnist_results(gans, epochs):
    res = {}
    for key, gan in gans.items():
        res[key] = {}
        print(key)
        print("\n\n\n")
        res[key]["mnist"] = {}
        train_iter, val_iter, test_iter = get_data(2000)
        if key == "vae":
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
        res[key]["mnist"]["KL-Divergence"] = kl
        res[key]["mnist"]["Jensen-Shannon"] = js
        res[key]["mnist"]["Wasserstein-Distance"] = wd
        res[key]["mnist"]["Energy-Distance"] = ed
        res[key]["mnist"]["DLoss"] = dl
        res[key]["mnist"]["GLoss"] = gl
    return res


def get_multivariate_graphs(res, gans, distance_metrics):
    for gan, value in gans.items():
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
    for gan, value in gans.items():
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
    for gan, value in gans.items():
        normal = pd.DataFrame(res[gan]['mnist'])
        for dist in distance_metrics:
            plt.plot(range(len(normal['mnist'])), normal['mnist'], label="MNIST")
            plt.xlabel("Epoch")
            plt.ylabel(dist)
            plt.title("{0}: {1}".format(gan.upper(), dist))
            plt.legend()
            plt.savefig('graphs/{0}_{1}.png'.format(gan, dist), dpi=100)
            plt.clf()
