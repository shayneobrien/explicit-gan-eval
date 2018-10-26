import sys, json, itertools, datetime
import matplotlib.pyplot as plt

import data
from utils import *
from models import w_gan, w_gp_gan, ns_gan, mm_gan, \
                   ls_gan, fisher_gan, ra_gan, info_gan, \
                   dra_gan, be_gan, vae, ae
from models.f_gan import forkl_gan, revkl_gan, tv_gan, \
                         js_gan, hellinger_gan, pearson_gan

plt.switch_backend('agg')

"""
    Choose \n
    (1) dataset: multivariate, mixture, circles, or mnist \n
    (2) trials (for confidence intervals) 1, 10, 100, etc. \n
    (3) number of dimensions: 1, 10, 100, 1000, etc. \n
    (4) number of epochs: 10, 100, 1000, etc. \n
    (5) number of samples: 1000, 10,000, 100,000, etc. \n
    (6) if choosing mixture, choose number of mixtures: 1, 10, 100, etc. \n
    e.g. python main.py multivariate 2 3 2 2
         python main.py mixture 2 3 2 10 10
         python main.py mnist 2 0 2 0
         python main.py circles 3 0 1 10
"""

if __name__ == "__main__":

    # Collect system args
    data_type = sys.argv[1]
    trials = int(sys.argv[2])
    dimensions = int(sys.argv[3]) # Data complexity
    epochs = int(sys.argv[4])
    samples = int(sys.argv[5])
    data_info = '{0}_dims_{1}_samples'.format(dimensions, samples)
    if data_type == "mixture":
        n_mixtures = int(sys.argv[6])
        data_info += '_{0}_mixtures'.format(n_mixtures)

    # Make output directories if they don't exist yet
    for dir in ['hypertuning', 'graphs', 'best', "confidence_intervals"]:
        for subdir in ['multivariate', 'mixture', 'circles', 'mnist']:
            dirname = dir + '/' + subdir + '/'
            if not os.path.exists(dirname):
                os.makedirs(dirname)

    # Set hyperparameters
    # Base learning rates for the smallest batch size (128). We will modify
    # these by a factor of 0.5 for each step up in batch size, as per
    # https://openreview.net/forum?id=B1Yy1BxCZ
    hidden_dims = [
                    32,
                    64,
                    128,
                    256,
                    512,
                    ]

    batch_sizes = [
                   128,
                   256,
                   512,
                   1024,
                   ]

    learning_rates = [
                      2e-1,
                      2e-2,
                      2e-3,
                      ]

    # Multivariate distributions
    distributions = [
                     'normal',
                     'beta',
                     'exponential',
                     'gamma',
                     'gumbel',
                     'laplace',
                     ]

    # Circles dataset
    modes = [
             2,
             4,
             8,
             16,
             ]

    n_circles = [
                 1,
                 2,
                 4,
                 8,
                 16,
                 ]

    # Specify models to test
    models = {
        "wgan": w_gan,
        "wgpgan": w_gp_gan,
        "nsgan": ns_gan,
        "lsgan": ls_gan,
        "mmgan": mm_gan,
        "dragan": dra_gan,
        "began": be_gan,
        "ragan": ra_gan,
        "infogan": info_gan,
        "fishergan": fisher_gan,
        "fgan_forward_kl": forkl_gan,
        "fgan_reverse_kl": revkl_gan,
        "fgan_jensen_shannon": js_gan,
        "fgan_total_var": tv_gan,
        "fgan_hellinger": hellinger_gan,
        "fgan_pearson": pearson_gan,
       # "vae": vae,
       # "autoencoder": ae,
    }

    # Distance metrics we will consider
    distance_metrics = [
                        "KL-Divergence",
                        "Jensen-Shannon",
                        "Wasserstein-Distance",
                        "Energy-Distance",
                        ]

    start_time = datetime.datetime.now().strftime("%Y-%m-%d-%s")
    out_dir = data_type + '/' + data_info + '/' + start_time

    for trial in range(1, trials+1):

        trial_path = 'hypertuning/' + out_dir + '/trial_{0}'.format(trial)
        if not os.path.exists(trial_path):
            os.makedirs(trial_path)

        print('========= TRIAL {0} ========= \n{1}'.format(trial, trial_path))

        for (lr, hdim, bsize) in itertools.product(*[learning_rates, hidden_dims, batch_sizes]):

            hyperparam = (lr * min(batch_sizes)/bsize, hdim, bsize)
            out_path = trial_path + '/results_{1}.json'.format(trial, "_".join([str(i) for i in hyperparam]))

            print('TRIAL: {0} | LR: {1} | HDIM: {2} | BSIZE: {3}' \
                .format(trial, hyperparam[0], hdim, bsize))

            if data_type == "multivariate":
                results = get_multivariate_results(models, distributions, dimensions,
                                                epochs, samples, hyperparam)

            elif data_type == "mixture":
                results = get_mixture_results(models, distributions, dimensions,
                                            epochs, samples, n_mixtures, hyperparam)

            elif data_type == "mnist":
                results = get_mnist_results(models, 784*1, # black and white, so 1 channel
                                          epochs, hyperparam)

            elif data_type == "circles":
                results = get_circle_results(models, 784*3, # RGB, so 3 channels
                                            epochs, samples, modes, n_circles, hyperparam)

            with open(out_path, 'w') as outfile:
                json.dump(results, outfile)

        # For each trial, get the best performing model with respect to hyperparameter setting.
        find_best = eval('get_best_performance_' + data_type)
        results = find_best(data_type, start_time, data_info, trial)

        # Output format is best/data_type/results_trial_time
        best_path =  'best/' + out_dir
        if not os.path.exists(best_path):
            os.makedirs(best_path)

        with open(best_path + '/trial_{1}.json'.format(data_type, trial), 'w') as outfile:
            json.dump(results, outfile)

    # Compute the confidence interval across the best results from each trial
    # TODO: Fix
#    get_ci = eval('get_confidence_intervals_' + data_type)
#    ci = get_ci(data_type, start_time, data_info)

    # Confidence interval filepath
#    ci_path = "confidence_intervals/{0}".format(out_dir)
#    if not os.path.exists(ci_path):
#        os.makedirs(ci_path)

#    with open(ci_path + '/data.json', 'w') as outfile:
#        json.dump(ci, outfile)
