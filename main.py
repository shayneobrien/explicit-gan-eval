import sys, json, itertools, datetime
import matplotlib.pyplot as plt

import data
from utils import *
from models import w_gan, w_gp_gan, ns_gan, mm_gan, \
                   ls_gan, fisher_gan, ra_gan, info_gan, \
                   dra_gan, be_gan

from models.f_gan import forkl_gan, revkl_gan, tv_gan, \
                         js_gan, hellinger_gan, pearson_gan

plt.switch_backend('agg')

"""
    Choose \n
    (1) dataset: multivariate \n
    (2) trials: 1, 5, 10, 20, etc. \n
    (3) number of dimensions: 1, 10, 100, 1000, etc. \n
    (4) number of epochs: 10, 100, 1000, etc. \n
    (5) number of samples: 1000, 10,000, 100,000, etc. \n

    e.g. \n
    python main.py multivariate 2 3 2 2
"""

if __name__ == "__main__":

    # Collect system args
    data_type = sys.argv[1]
    trials = int(sys.argv[2])
    dimensions = int(sys.argv[3])
    epochs = int(sys.argv[4])
    samples = int(sys.argv[5])
    data_info = '{0}_dims_{1}_samples'.format(dimensions, samples)

    # Make output directories if they don't exist yet
    for dir in ['hypertuning', 'graphs', 'best', "confidence_intervals"]:
        dirname = dir + '/multivariate/'
        if not os.path.exists(dirname):
            os.makedirs(dirname)

    # Set hyperparameters
    hidden_dims = [
                    16,
                    32,
                    64,
                    128,
                    ]
    batch_sizes = [
                   # 128,
                   # 256,
                   # 512,
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

    # Distance metrics we will consider
    distance_metrics = [
                        "KL-Divergence",
                        "Jensen-Shannon",
                        "Wasserstein-Distance",
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
        "fgan_pearson": pearson_gan}

    start_time = datetime.datetime.now().strftime("%Y-%m-%d-%s")
    out_dir = data_type + '/' + data_info + '/' + start_time

    for trial in range(1, trials+1):

        trial_path = 'hypertuning/' + out_dir + '/trial_{0}'.format(trial)
        if not os.path.exists(trial_path):
            os.makedirs(trial_path)

        print('========= TRIAL {0} ========= \n{1}'.format(trial, trial_path))

        for (lr, hdim, bsize) in itertools.product(*[learning_rates, hidden_dims, batch_sizes]):

            # Modify learning rate by a factor of 0.5 for each step up in batch size, as per
            # https://openreview.net/forum?id=B1Yy1BxCZ
            # NOTE: this is not done in the paper; we only consider batch size 1024
            hyperparam = (lr * min(batch_sizes)/bsize, hdim, bsize)

            # Output path for the given hyperparameter setting
            out_path = trial_path + '/results_{1}.json'.format(trial, "_".join([str(i) for i in hyperparam]))

            # For logging
            print('TRIAL: {0} | LR: {1} | HDIM: {2} | BSIZE: {3}' \
                .format(trial, hyperparam[0], hdim, bsize))

            # Train model, compute divergences, log everything
            results = get_multivariate_results(models, distributions, dimensions,
                                            epochs, samples, hyperparam)

            # Write results to output file
            with open(out_path, 'w') as outfile:
                json.dump(results, outfile)

        find_best = eval('get_best_performance_' + data_type)
        results = find_best(data_type, start_time, data_info, trial)

        # Output format is best/data_type/results_trial_time
        best_path =  'best/' + out_dir
        if not os.path.exists(best_path):
            os.makedirs(best_path)

        # Write out best results from the current trial
        with open(best_path + '/trial_{1}.json'.format(data_type, trial), 'w') as outfile:
            json.dump(results, outfile)
