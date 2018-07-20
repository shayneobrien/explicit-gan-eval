import os, sys, json, itertools
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset

import data
from models import w_gan, w_gp_gan, ns_gan, mm_gan, ls_gan, dra_gan, be_gan, vae, autoencoder
from utils import *

plt.switch_backend('agg')
plt.style.use('fivethirtyeight')


if __name__ == "__main__":
    print("""
        Choose \n
        (1) dataset: multivariate, mixture, circles, or mnist \n
        (2) number of dimensions: 1, 10, 100, 1000, etc. \n
        (3) number of epochs: 10, 100, 1000, etc. \n
        (4) number of samples: 1000, 10,000, 100,000, etc. \n
        (5) if choosing mixture, choose number of mixtures: 1, 10, 100, etc. \n
        e.g. python main.py multivariate 200 15 10000 \n
             python main.py mixture 3 5 10 10
        """)
    data_type = sys.argv[1]
    dimensions = int(sys.argv[2])
    epochs = int(sys.argv[3])
    samples = int(sys.argv[4])
    if data_type == "mixture":
        print("adding n_mixtures")
        n_mixtures = int(sys.argv[5])

    # Make output directories if they don't exist yet
    for dir in ['hypertuning', 'graphs']:
        for subdir in ['multivariate', 'mixture', 'circles', 'mnist']:
            dirname = dir + '/' + subdir + '/'
            if not os.path.exists(dirname):
                os.makedirs(dirname)


    # Set hyperparameters
    learning_rates = [1e-3, 5e-4, 1e-4, 5e-5]
    hidden_dims = [16, 32, 64, 128, 256]
    BATCH_SIZE = [100, 150, 200, 250]
    distributions = ['normal',
                     'beta',
                     'exponential',
                     # 'gamma', # TODO: Fix gamma
                     'gumbel',
                     'laplace']

    models = {
        # "wgan": w_gan,
        # "wgpgan": w_gp_gan,
        # "nsgan": ns_gan,
        # "lsgan": ls_gan,
        # "mmgan": mm_gan,
        # "dragan": dra_gan,
        # "began": be_gan,
        # "vae": vae,
        "autoencoder": autoencoder,
    }

    distance_metrics = ["KL-Divergence", "Jensen-Shannon", "Wasserstein-Distance", "Energy-Distance",]
    for hyperparam in list(itertools.product(*[learning_rates, hidden_dims, BATCH_SIZE])):
        out_path = 'hypertuning/' + data_type + '/data{0}.json'.format(str(hyperparam))
        if data_type == "multivariate":
            results = get_multivariate_results(models, distributions, dimensions,
                                            epochs, samples, hyperparam)
            # We only want to graph the best ones!
            # get_multivariate_graphs(results, models, distributions, distance_metrics, epochs)
        elif data_type == "mixture":
            # TODO: output results
            results = get_mixture_results(models, distributions, dimensions,
                                        epochs, samples, n_mixtures, hyperparam)
            get_mixture_graphs(results, models, distributions, distance_metrics, epochs)
        # elif data_type == "circles":
        #     # TODO: Graphing circles
        #     results = get_circle_results(models, dimensions, epochs, samples)
        # elif data_type == "mnist":
        #     # TODO: Graphing MNIST (see VAE code)
        #     results = get_mnist_results(models, epochs)
            # get_mnist_graphs(res, models, distance_metrics)

        with open(out_path, 'w') as outfile:
            json.dump(results, outfile)
    print("Le Fin.")
