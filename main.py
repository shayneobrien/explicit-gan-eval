import os, sys, json, itertools, datetime
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset

import data
from models import w_gan, w_gp_gan, ns_gan, mm_gan, ls_gan, dra_gan, be_gan, vae, autoencoder
from utils import *

plt.switch_backend('agg')
# plt.style.use('fivethirtyeight') # Let's style things later


if __name__ == "__main__":
    print("""
        Choose \n
        (1) dataset: multivariate, mixture, circles, or mnist \n
        (2) trials (for confidence intervals) 1 \n
        (3) number of dimensions: 1, 10, 100, 1000, etc. \n
        (4) number of epochs: 10, 100, 1000, etc. \n
        (5) number of samples: 1000, 10,000, 100,000, etc. \n
        (6) if choosing mixture, choose number of mixtures: 1, 10, 100, etc. \n
        e.g. python main.py multivariate 3 3 3 3 \n
             python main.py mixture 3 3 5 10 10
        """)
    data_type = sys.argv[1]
    trials = int(sys.argv[2])
    dimensions = int(sys.argv[3])
    epochs = int(sys.argv[4])
    samples = int(sys.argv[5])
    if data_type == "mixture":
        print("adding n_mixtures")
        n_mixtures = int(sys.argv[6])

    # Make output directories if they don't exist yet
    for dir in ['hypertuning', 'graphs', 'best', "confidence_intervals"]:
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
        "wgan": w_gan,
        # "wgpgan": w_gp_gan,
        # "nsgan": ns_gan,
        # "lsgan": ls_gan,
        # "mmgan": mm_gan,
        # "dragan": dra_gan,
        # "began": be_gan,
        "vae": vae,
        "autoencoder": autoencoder,
    }

    distance_metrics = ["KL-Divergence", "Jensen-Shannon", "Wasserstein-Distance", "Energy-Distance"]
    for t in range(trials):
        print('Trial {0}'.format(t))
        for hyperparam in list(itertools.product(*[learning_rates, hidden_dims, BATCH_SIZE])):
            print(hyperparam)
            out_path = 'hypertuning/' + data_type + '/results_{0}.json'.format("_".join([str(i) for i in hyperparam]))
            if data_type == "multivariate":
                results = get_multivariate_results(models, distributions, dimensions,
                                                epochs, samples, hyperparam)
                # get_multivariate_graphs()
            elif data_type == "mixture":
                # TODO: output results
                results = get_mixture_results(models, distributions, dimensions,
                                            epochs, samples, n_mixtures, hyperparam)
                # get_mixture_graphs(results, models, distributions, distance_metrics, epochs)
            elif data_type == "circles":
                pass
                # TODO: Graphing circles
                # results = get_circle_results(models, dimensions, epochs, samples)
            elif data_type == "mnist":
                pass
                # TODO: Graphing MNIST (see VAE code)
                # results = get_mnist_results(models, epochs)
                # get_mnist_graphs(res, models, distance_metrics)
            with open(out_path, 'w') as outfile:
                json.dump(results, outfile)
        results = get_best_performance(data_type)
        with open("best/{}/results_{}_{}.json".format(data_type, t, datetime.datetime.now().strftime("%Y-%m-%d")), 'w') as outfile:
                json.dump(results, outfile)
    ci = get_confidence_intervals(data_type)
    with open("confidence_intervals/{}/data.json".format(data_type), 'w') as outfile:
        json.dump(ci, outfile)
    # get_best_graph(results, models, distributions, distance_metrics, epochs)
    print("Le Fin.")
