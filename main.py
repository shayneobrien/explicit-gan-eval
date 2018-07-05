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
from .utils import *

if __name__ == "__main__":

    # hyperparam = [5e-3, 256, 5, 10]
    learning_rates = [1e-3, 5e-4, 1e-4, 5e-5]
    hidden_dims = [16, 32, 64, 128, 256]
    BATCH_SIZE = [100, 150, 200, 250]

    print("Choose a dataset: multivariate, mixture, circles, or mnist")
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
    distributions = ['normal']
    gans = [wgan, wgpgan, nsgan, lsgan, mmgan, nsgan, dragan, began, vae]
    gans_index = ["wgan", "wgpgan", "nsgan", "lsgan", "mmgan", "nsgan", 'dragan', "began", "vae"]
    # gans = [wgan, vae]
    # gans_index = ["wgan", "vae"]
    distance_metrics = ["KL-Divergence", "Jensen-Shannon", "Wasserstein-Distance", "Energy-Distance"]
    if data_type == "multivariate":
        for hyperparam in list(itertools.product(*[learning_rates, hidden_dims, BATCH_SIZE])):
            lr, dim, bsize = hyperparam
            print(hyperparam)
            res = get_multivariate_results(gans, gans_index, distributions, dimensions, epochs, samples, hyperparam)
            print(type(res))
            with open('hypertuning/data{0}.json'.format(str(hyperparam)), 'w') as outfile:
                json.dump(res, outfile)
        # get_multivariate_graphs(res, gans_index, distance_metrics)
    elif data_type == "mixture":
        res = get_mixture_results(gans, gans_index, distributions, dimensions, epochs, samples, n_mixtures)
        get_mixture_graphs(res, gans_index, distance_metrics)
    elif data_type == "circles":
        res = get_circle_results(gans, gans_index, dimensions, epochs, samples)
        print("to do: graph circles")
    elif data_type == "mnist":
        res = get_mnist_results(gans, gans_index, epochs)
        with open('mnistgoodies.json', 'w') as outfile:
                json.dump(res, outfile)
        get_mnist_graphs(res, gans_index, distance_metrics)

        print("to do: graph circles")
    print("Le Fin.")
