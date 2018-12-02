import os, shutil, sys, json, itertools
import torch
import data
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
from functools import partial
from itertools import product
from tabulate import tabulate
from collections import defaultdict
from tqdm import tqdm, tqdm_notebook
from models.model_utils import preprocess, compute_divergences
from IPython.core.display import display, HTML

from data import *
from models import w_gan, w_gp_gan, ns_gan, mm_gan, \
                   ls_gan, fisher_gan, ra_gan, info_gan, \
                   dra_gan, be_gan

from models.f_gan import forkl_gan, revkl_gan, tv_gan, \
                         js_gan, hellinger_gan, pearson_gan

plt.switch_backend('agg')

"""
Results
"""

def get_multivariate_results(models, distributions, dimensions,
                             epochs, samples, hyperparameters):
    """ Multivariate distribution results """
    results, activation_type = nested_pickle_dict(), 'relu'
    lr, hdim, bsize = hyperparameters

    for idx, (model_name, module) in enumerate(models.items()):
        model = module.Model(image_size=dimensions,
                             hidden_dim=hdim,
                             z_dim=int(round(max(hdim/4, 1))),
                             atype=activation_type)

        for dist in distributions:

            print('\n', model_name, dist, 'MULTIVARIATE', idx, '/', len(models.keys()))
            gen = data.Distribution(dist_type=dist, dim=dimensions)
            metrics = model_results(module, epochs, hyperparameters,
                                    gen, samples, dimensions, activation_type)
            results[model_name][dist].update(metrics)

    return results


def model_results(module, epochs, hyperparameters, gen, samples, dimensions, activation_type):
    """ Train a model, get metrics dictionary out """

    # Unpack hyperparameters
    lr, hdim, bsize = hyperparameters

    # Create data iterators
    train_iter, test_iter = preprocess(gen, samples, bsize, epochs)

    # Init model
    model = module.Model(image_size=dimensions,
                         hidden_dim=hdim,
                         z_dim=int(round(max(hdim/4, 1))),
                         atype=activation_type)

    # Init trainer
    trainer = module.Trainer(model=model,
                             train_iter=train_iter,
                             val_iter=None,
                             test_iter=test_iter)

    # Train and get output metrics
    metrics = trainer.train(num_epochs=epochs,
                            lr=lr)

    return metrics


"""
"Best" results for a given trial according to minimum performance with respect to
tested hyperparameters for that trial
"""


def crawl_directory(dirname):
    """ Walk a nested directory to get all filename ending in a pattern """
    for path, subdirs, files in os.walk(dirname):
        for name in files:
            if not name.endswith('.DS_Store'):
                yield os.path.join(path, name)


def remove_empty_dirs(path):
    for root, dirnames, filenames in os.walk(path, topdown=False):
        for dirname in dirnames:
            remove_empty_dir(os.path.realpath(os.path.join(root, dirname)))


def remove_empty_dir(path):
    try:
        os.rmdir(path)
    except OSError:
        pass1


def nested_pickle_dict():
    """ Picklable defaultdict nested dictionaries """
    return defaultdict(nested_pickle_dict)


def format_e(n):
    a = '%E' % n
    return (a.split('E')[0].rstrip('0').rstrip('.') + 'E' + a.split('E')[1]).lower()


def mean_confidence_interval(data, axis=0, confidence=0.95):
    n = data.shape[axis]
    mu, std = np.nanmean(data, axis=axis), scipy.stats.sem(data, axis=axis, nan_policy='omit')
    h = np.ma.getdata(std) * scipy.stats.t.ppf((1 + confidence) / 2., n-1)

    return mu, h, mu-h, mu+h

def load_best(filename):
    with open(filename, 'r') as f:
        data = json.load(f)

    return data


def get_best_multivariate(dirname='../hypertuning/multivariate/'):
    """ Find best results """

    # Get filenames
    dim_numsamples_names = [i for i in os.listdir(dirname) if i != '.DS_Store']
    lr_hdim_bsize_names = [i for i in os.listdir(dirname + dim_numsamples_names[0] + '/trial_1')
                           if '1024' in i]

    # For each number of dimensions and training samples
    for name in tqdm_notebook(dim_numsamples_names):

        # Initialize best dictionary
        best_path = '../best/multivariate/{0}/'.format(t)
        global_optimal = nested_pickle_dict()

        # For each hyperparameter setting
        for t in tqdm_notebook(lr_hdim_bsize_names):

            # Initialize a dictionary containing the best result
            optimal = nested_pickle_dict()
            results = []

            # Load in the results from each trial
            for trial in range(1, 21):
                path = dirname + '{0}/trial_{1}/{2}'.format(name, trial, t)

                data = []
                with open(path) as f:
                    for line in f:
                        data.append(json.loads(line))

                results.append(data[0])

            # Append the results through each model, distribution, and metric
            for result in results:
                for model, distributions in result.items():
                    for distribution, metrics in distributions.items():
                        for metric, values in metrics.items():
                            if metric in ["LR", "HDIM", 'GLoss', 'DLoss', "BSIZE", "Energy-Distance"]:
                                continue
                            else:
                                # If metric is seen for the first time, initialize it
                                if 'values' not in optimal[model][distribution][metric]:
                                    optimal[model][distribution][metric]["values"] = []

                                # Otherwise, append it
                                optimal[model][distribution][metric]["values"].append(values)

            # Go back through each model, distribution, and metric and find the best performing result
            # according to minimum performance
            for model, distributions in result.items():
                for distribution, metrics in distributions.items():
                    for metric, values in metrics.items():
                        if metric in ["LR", "HDIM", 'GLoss', 'DLoss', "BSIZE", "Energy-Distance"]:
                            continue
                        else:
                            data_min = np.nanmean(np.nanmin(np.array(optimal[model][distribution][metric]["values"]), axis=1))

                            # Init global optimal
                            if 'best' not in global_optimal[model][distribution][metric]:
                                global_optimal[model][distribution][metric]['best'] = 1e10

                            # If current min is less than the global best, update it and compute confidence intervals.
                            if data_min < global_optimal[model][distribution][metric]['best']:
                                global_optimal[model][distribution][metric]['best'] = data_min
                                global_optimal[model][distribution][metric]['parameters'] = [metrics["LR"], metrics["HDIM"], metrics["BSIZE"]]
                                global_optimal[model][distribution][metric]["values"] = optimal[model][distribution][metric]["values"]

                                mean, h, low, high = mean_confidence_interval(np.array(optimal[model][distribution][metric]["values"]), axis=0)

                                global_optimal[model][distribution][metric]['low'] = list(low)
                                global_optimal[model][distribution][metric]['h'] = list(h)
                                global_optimal[model][distribution][metric]['mean'] = list(mean)
                                global_optimal[model][distribution][metric]['high'] = list(high)

        # Initialize best path and dump results to '../best/multivariate/'
        if not os.path.exists(best_path):
            os.makedirs(best_path)

        with open(best_path + 'data.json', 'w') as outfile:
            json.dump(global_optimal, outfile)


def get_best_per_trial(mypath):
    """ For a trial, get the best performance for multivariate data according to any hyperparam """
    # Get path, files in path
    files = os.listdir(mypath)
    results = []

    # Read in the files
    for file in files:
        if file == '.DS_Store':
            continue

        with open(mypath + file, 'r') as f:
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
                        elif min(optimal[model][distribution][metric]["value"]) > min(values):
                            optimal[model][distribution][metric]["value"] = values
                            optimal[model][distribution][metric]["parameters"] = [metrics["LR"], metrics["HDIM"], metrics["BSIZE"]]

    return optimal


def multivariate_hypertuning2best(dirname='/Users/sob/Desktop/gan_results/hypertuning/multivariate/64_dims_100000_samples/'):
    """ Move HYPERTUNING RESULTS TO BEST FOLDER """
    best_path = '../best/' + '/'.join(dirname.split('/')[-3:])
    if not os.path.exists(best_path):
        os.makedirs(best_path)

    files = os.listdir(dirname)
    files = [f for f in files if f != '.DS_Store']
    for idx, f in tqdm.tqdm_notebook(enumerate(files)):

        optimal = get_best_per_trial(dirname + f + '/')
        if len(os.listdir(dirname + f + '/')) < 60:
            print(f, len(os.listdir(dirname + f + '/')))

        with open(best_path + '/trial_{0}.json'.format(idx+1), 'w') as outfile:
            json.dump(optimal, outfile)


def merge_multivariate(dirname):
    """ Merge multivariate results from parallelized jobs into a single folder
    (warning: not recommended to run this function multiple times in a row)"""
    outdir = dirname
    for idx, file in enumerate(os.listdir(dirname)):

        if '.DS_Store' in file:
            continue

        for nest in crawl_directory(dirname + file):

            index = 1

            if 'dims' not in nest.split('/')[6]:
                outdir = '/'.join(nest.split('/')[:7] + nest.split('/')[8:9]) + '/'
            else:
                # Uncomment the + for mixture
                outdir = dirname + nest.split('/')[6] + '/'

            # Initialize directory
            if not os.path.exists(outdir + 'trial_{0}/'.format(index)):
                os.makedirs(outdir + 'trial_{0}/'.format(index))

            try:
                shutil.move(nest, outdir + 'trial_{0}/'.format(index))
            except:
                extension = nest.split('/')[-1]
                while os.path.exists(outdir + 'trial_{0}/'.format(index) + extension):
                    index += 1

                if not os.path.exists(outdir + 'trial_{0}/'.format(index)):
                    os.makedirs(outdir + 'trial_{0}/'.format(index))

                shutil.move(nest, outdir + 'trial_{0}/'.format(index))

    remove_empty_dirs(dirname)


def identify_failed_trials(dirname='../hypertuning/multivariate/'):
    """ Get missing runs for all trials due to occasional run failure due to GAN instability """
    hidden_dims = [32, 64, 128, 256, 512]
    batch_sizes = [128, 256, 512, 1024]
    learning_rates = [2e-1, 2e-2, 2e-3]

    filenames, hyperparams = [], []

    for (lr, hdim, bsize) in product(*[learning_rates, hidden_dims, batch_sizes]):
        hyperparam = (lr * min(batch_sizes)/bsize, hdim, bsize)
        filename = 'results_{0}.json'.format("_".join([str(i) for i in hyperparam]))
        filenames.append(filename)
        hyperparams.append((str(format_e(lr)), str(hdim), str(bsize)))

    TODO = []
    for file in os.listdir(dirname):
        if '.DS_Store' in file:
            continue

        print(file, len(os.listdir(dirname + file)))
        idx = 0
        try:
            for f in os.listdir(dirname + file):
                if '.DS_Store' in f:
                    continue

                files = os.listdir(dirname + file + '/' + f)
                length = len(files)
                print(f, length)

                if length >= 60:
                    idx += 1
                else:
                    missing = [hyperparams[idx] for idx, item in enumerate(filenames) if item not in files]
                    TODO.extend(missing)

            print('{0}/20'.format(idx))
            print('\n')

        except NotADirectoryError:
            files = os.listdir(dirname + file)
            missing = [hyperparams[idx] for idx, item in enumerate(filenames) if item not in files]
            TODO.extend(missing)


    return TODO


"""
VISUALIZATION: Reproducing tables and figures
"""
# Some styling
plt.rcParams['axes.axisbelow'] = True

# For plotting, indexing models
model_names = ["wgan", "wgpgan", "nsgan", "mmgan", "ragan",
               "lsgan", "dragan", "began", "infogan", "fishergan",
               "fgan_forward_kl", "fgan_reverse_kl", "fgan_jensen_shannon",
               "fgan_total_var", "fgan_hellinger", "fgan_pearson"]
plot_names = ['WGAN', 'WGANGP', 'NSGAN', 'MMGAN', 'RAGAN', 'LSGAN', 'DRAGAN', 'BEGAN', 'InfoGAN',
              'FisherGAN','ForwGAN', 'RevGAN', 'JSGAN', 'TVGAN', 'HellingerGAN', 'PearsonGAN', 'Expected']
distance_metrics=["KL-Divergence", "Jensen-Shannon", "Wasserstein-Distance"]#, "Energy-Distance"]
title_names=["Kullback-Leibler Divergence", "Jensen-Shannon Divergence", "Wasserstein Distance"]
distributions=['normal', 'beta', 'gumbel', 'laplace', 'exponential', 'gamma']

# Colors
palette = ["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a", "#d62728", "#ff9896", "#9467bd",
            "#c5b0d5", "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7", "#bcbd22" "#dbdb8d",
            "#17becf", "#9edae5"]

# For subplotting
plt_idx = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

def graph_multivariate(expected=None, ci=True, omit_outliers=True, thresh=8.5, save=False):
    """ Print performance as a function of epoch for best performing hyperparameter

    Reproduces subplots from Figures 1, 2, and 3
    """

    if expected is None:
        expected = get_empirical_divergences()

    for d in [16, 32, 64, 128]:
        for s in [1000, 10000, 100000]:

            optimal = load_json('../best/multivariate/{0}_dims_{1}_samples/data.json'.format(d, s))

            for m_idx, metric in enumerate(distance_metrics):

                fig, axes = plt.subplots(2, 3, sharex=True, sharey=False, figsize=(20,20))

                for d_idx, distribution in enumerate(distributions):

                    # For controlling y-axis limit
                    max_upper = 0

                    for idx, model_name in enumerate(model_names):


                        if model_name in ['vae', 'autoencoder']:
                            continue

                        # Mean minimum performance
                        data = optimal[model_name][distribution][metric]['mean']

                        if omit_outliers:
                            # Remove outliers
                            orig_data = np.array(data)
                            notnan_indexes = ~np.isnan(orig_data)
                            outlier_indexes = is_outlier(orig_data[notnan_indexes], thresh=thresh)
                            data = orig_data.flatten()[~outlier_indexes]

                            x = np.linspace(0, orig_data.shape[0], data.shape[0])
                            high = data + np.array(optimal[model_name][distribution][metric]['h']).flatten()[~outlier_indexes]
                            low = data - np.array(optimal[model_name][distribution][metric]['h']).flatten()[~outlier_indexes]

                        else:

                            x = range(0, len(data))
                            high = data + np.array(optimal[model_name][distribution][metric]['h'])
                            low = data - np.array(optimal[model_name][distribution][metric]['h'])

                        axes[plt_idx[d_idx][0], plt_idx[d_idx][1]].plot(x,
                                                                        data,
                                                                        label=plot_names[idx],
                                                                        c=palette[idx])
                        maxi = max(data) + (0.10*max(data))
                        if maxi > max_upper:
                            max_upper = maxi

                        if ci == True:
                            axes[plt_idx[d_idx][0], plt_idx[d_idx][1]].fill_between(x,
                                                                                    high,
                                                                                    [max(0, i) for i in low],
                                                                                    facecolor=palette[idx],
                                                                                    alpha=0.09)

                    axes[plt_idx[d_idx][0], plt_idx[d_idx][1]].grid(alpha=0.3)
                    axes[plt_idx[d_idx][0], plt_idx[d_idx][1]].set_xlim(0, 25)
                    axes[plt_idx[d_idx][0], plt_idx[d_idx][1]].set_ylim(top=max_upper, bottom=0)
                    axes[plt_idx[d_idx][0], plt_idx[d_idx][1]].set_title("{0}".format(distribution.capitalize()),
                                                                        fontsize=18)
                    axes[plt_idx[d_idx][0], plt_idx[d_idx][1]].plot(x, [expected[distribution][metric][d]]*len(x),
                                                                    'k--',
                                                                    label='Expected')

                fig.suptitle("{0}: {1}-dimensions {2}-samples".format(title_names[m_idx], d, s), x=0.5, y=0.93, fontsize=20)
                fig.text(0.5, 0.08, "Epoch", ha='center', fontsize=18)
                plt.legend(loc='center left', bbox_to_anchor=(1, 1), fontsize=20)
                if save == True:
                    plt.savefig('../graphs/multivariate/{0}_{1}_{2}.png'.format(metric, d, s),
                                dpi=100, bbox_inches='tight')

                plt.show()


def graph_fncsamples(param_dict, expected=None, save=False):
    """ Plot mean minimum performance with error bars as a function of number samples

    Reproduce subplots from Figures 4, 5, 6
    """

    if expected is None:
        expected = get_empirical_divergences()

    samples = [1000, 10000, 100000]

    for dims in [16, 32, 64, 128]:

        for metric in distance_metrics:

            fig, axes = plt.subplots(2, 3, sharex=True, sharey=False, figsize=(15,15))

            for d_idx, distribution in enumerate(distributions):

                for idx, model_name in enumerate(model_names):

                    x, y, yerr = [], [], []

                    for s_idx, val in enumerate(param_dict[metric][model_name][distribution][dims].split('\n')):

                        mu, err = val.split('±')
                        x.append(float(samples[s_idx])), y.append(float(mu)), yerr.append(float(err))

                    axes[plt_idx[d_idx][0], plt_idx[d_idx][1]].errorbar(x, y, xerr=0, yerr=yerr,
                                                                          barsabove=True, label=plot_names[idx],
                                                                          c=palette[idx], ecolor=palette[idx])

                    axes[plt_idx[d_idx][0], plt_idx[d_idx][1]].set_title("{0}".format(distribution.capitalize()))

                axes[plt_idx[d_idx][0], plt_idx[d_idx][1]].grid(alpha=0.5)
                axes[plt_idx[d_idx][0], plt_idx[d_idx][1]].plot(x, [expected[distribution][metric][dims]]*len(x),
                                                                        'k--',
                                                                        label=plot_names[-1])

            plt.xscale('log')
            fig.suptitle("{0} {1}-dims".format(metric, dims), x=0.5, y=0.93, fontsize=18)
            fig.text(0.5, 0.07, "Number samples", ha='center', fontsize=15)
            plt.legend(loc='center left', bbox_to_anchor=(1, 1), fontsize=15)
            if save:
                plt.savefig('../graphs/multivariate/samplesfnc_{0}_{1}_.png'.format(metric, dims), dpi=100)
            plt.show()


def print_confidence_intervals():
    """ Print confidence intervals for minimum across all runs """

    rankings_dict = nested_pickle_dict()
    param_dict = nested_pickle_dict()

    # Cycle through settings
    for dims in [16, 32, 64, 128]:

        for samples in [1000, 10000, 100000]:

            # Load
            optimal = load_json('../best/multivariate/{0}_dims_{1}_samples/data.json'.format(dims, samples))

            for metric in distance_metrics:

                for distribution in distributions:

                    for model_name in model_names:

                        # Find confidence intervals
                        minimums = np.nanmin(np.array(optimal[model_name][distribution][metric]['values']), axis=1)
                        mu, h, low, high = mean_confidence_interval(minimums)

                        # Save to dictionary
                        if dims not in param_dict[metric][model_name][distribution]:
                            param_dict[metric][model_name][distribution][dims] = '%.3f ± %.3f' % (np.round(mu, 3), np.round(h, 3))
                        else:
                            param_dict[metric][model_name][distribution][dims] += '\n%.3f ± %.3f' % (np.round(mu, 3), np.round(h, 3))


    for dims in [16, 32, 64, 128]:

        for samples in [1000, 10000, 100000]:

            for metric in distance_metrics:

                print(metric, dims)

                data = [[key, param_dict[metric][key]['normal'][dims], param_dict[metric][key]['beta'][dims],
                              param_dict[metric][key]['gumbel'][dims], param_dict[metric][key]['laplace'][dims],
                              param_dict[metric][key]['exponential'][dims], param_dict[metric][key]['gamma'][dims]]
                         for key in model_names]

                print(tabulate(data, headers=['Model', 'Normal', 'Beta', 'Gumbel', 'Laplace', 'Exponential', 'Gamma'], tablefmt='fancy_grid'), '\n')

    return param_dict


def get_trainable_param_counts():
    """ Counter number of trainable parameters for each model """

    models = {
        "wgan": w_gan, "wgpgan": w_gp_gan, "nsgan": ns_gan, "lsgan": ls_gan, "mmgan": mm_gan,
        "dragan": dra_gan, "began": be_gan, "ragan": ra_gan, "infogan": info_gan, "fishergan": fisher_gan,
        "fgan_forward_kl": forkl_gan, "fgan_reverse_kl": revkl_gan, "fgan_jensen_shannon": js_gan,
        "fgan_total_var": tv_gan, "fgan_hellinger": hellinger_gan, "fgan_pearson": pearson_gan,
    }

    for hdim in [32, 64, 128, 256, 512]:
        for dimensions in [16, 32, 64, 128]:
            print('Hidden dim: {0} | Data dim: {1}'.format(hdim, dimensions))
            for idx, (model_name, module) in enumerate(models.items()):
                model = module.Model(image_size=dimensions,
                                     hidden_dim=hdim,
                                     z_dim=int(round(max(hdim/4, 1))),
                                     atype='relu')

                print(model_name, count_parameters(model))

            print('\n')


def print_best_hyperparameters():
    """ Print best performing hyperparameters in LaTeX format
    (first row = 1k samples, second = 10k, third = 100k) """

    # Cycle through settings
    for dims in [16, 32, 64, 128]:

        for samples in [1000, 10000, 100000]:

            # Load
            optimal = load_json('../best/multivariate/{0}_dims_{1}_samples/data.json'.format(dims, samples))
            print('==========={0}-dims-{1}-samples==========='.format(dims, samples))

            for idx, key in enumerate(model_names):

                print(plot_names[idx], end="", flush=True)
                for metric in distance_metrics:
                    print(metric)
                    for dist in distributions:

                        print(' & ', optimal[key][dist][metric]['parameters'][:2], end="", flush=True)
                    print(" \\ \midrule ", end='')

                print('\n')

        print('\n\n\n\n')


def hyperparameter_robustness(dirname='../best/multivariate/'):
    """ Find number of other settings within its confidence interval """

    # Get filenames
    dim_numsamples_names = [i for i in os.listdir('../hypertuning/multivariate/') if i != '.DS_Store']
    lr_hdim_bsize_names = [i for i in os.listdir('../hypertuning/multivariate/' + dim_numsamples_names[0] + '/trial_1')
                           if '1024' in i]

    # Dict for counting number of hyperparams within global min's confidence interval
    robust = nested_pickle_dict()

    for name in tqdm_notebook(dim_numsamples_names):

        print('Loading {0}...'.format(name))

        global_optimal = load_json(dirname + '{0}/data.json'.format(name))

        # Initialize best dictionary
        for t in lr_hdim_bsize_names:

            optimal = nested_pickle_dict()
            results = []

            # Load in the results from each trial
            for trial in range(1, 21):
                path = '../hypertuning/multivariate/{0}/trial_{1}/{2}'.format(name, trial, t)

                data = []
                with open(path) as f:
                    for line in f:
                        data.append(json.loads(line))

                results.append(data[0])

            # Go through each one and append the results
            for result in results:
                for model, distributions in result.items():
                    for distribution, metrics in distributions.items():
                        for metric, values in metrics.items():
                            if metric in ["LR", "HDIM", 'GLoss', 'DLoss', "BSIZE", "Energy-Distance"]:
                                continue
                            else:

                                # If metric is seen for the first time, initialize it
                                if 'values' not in optimal[model][distribution][metric]:
                                    optimal[model][distribution][metric]["values"] = []

                                # Otherwise, compare it the presently considered value
                                optimal[model][distribution][metric]["values"].append(values)

            # Go through each one to count the number of hyperparameters with performances that fall
            # into the best average minimum performance
            for model, distributions in result.items():
                for distribution, metrics in distributions.items():
                    for metric, values in metrics.items():
                        if metric in ["LR", "HDIM", 'GLoss', 'DLoss', "BSIZE", "Energy-Distance"]:
                            continue
                        else:

                            # Initialize
                            if metric not in robust[model][name][distribution]:
                                robust[model][name][distribution][metric] = 0

                            if 'total' not in robust[model]['all']:
                                robust[model]['all']['total'] = 0

                            if metric not in robust[model]['all']:
                                robust[model]['all'][metric] = 0

                            # Find global min compared to current data min
                            _, _, global_low, global_high = mean_confidence_interval(np.array(global_optimal[model][distribution][metric]['mean']))
                            data_mean, _, data_low, data_high = mean_confidence_interval(np.nanmin(np.array(optimal[model][distribution][metric]["values"]), axis=1))

                            # If it's within the global min confidence interval, increment
                            if global_low <= data_mean <= global_high:

                                robust[model][name][distribution][metric] += 1
                                robust[model]['all']['total'] += 1
                                robust[model]['all'][metric] += 1

    # Print results
    for i in robust.keys():
        print(i)
        for k in ['KL-Divergence','Jensen-Shannon', 'Wasserstein-Distance']:
            if k == 'total':
                continue
            print(k, robust[i]['all'][k])
        print('TOTAL:',robust[i]['all']['total'], (robust[i]['all']['total']/51840) * 100)
        print('\n')

    return robust


def get_empirical_divergences(output=False):
    """ How the 'Expected' dashed lines were produced in Figures 1-6 """
    print('Finding expected empirical divergences...')
    expected = nested_pickle_dict()

    for dist in tqdm_notebook(['normal', 'beta', 'gumbel', 'laplace', 'exponential', 'gamma']):

        for dims in [16, 32, 64, 128]:

            gen = Distribution(dist_type=dist, dim=dims)

            for j in range(20):

                A = gen.generate_samples(1024)
                B = gen.generate_samples(1024)
                results = compute_divergences(np.array(A), B)

                for i in results.keys():
                    if dims not in expected[dist][i]:
                        expected[dist][i][dims] = []
                    expected[dist][i][dims].append(results[i])


    for metric in ['KL-Divergence', 'Jensen-Shannon', 'Wasserstein-Distance']:
        for dims in [16, 32, 64, 128]:
            for dist in ['normal', 'beta', 'gumbel', 'laplace', 'exponential', 'gamma']:
                expected[dist][metric][dims] = np.mean(expected[dist][metric][dims])
                if output:
                    print(dist, metric, dims, np.mean(expected[dist][metric][dims]))

    return expected


def mean_confidence_interval(data, axis=0, confidence=0.95):
    """ Compute confidence intervals """
    n = data.shape[axis]

    mu, std = np.nanmean(data, axis=axis), scipy.stats.sem(data, axis=axis, nan_policy='omit')
    h = np.ma.getdata(std) * scipy.stats.t.ppf((1 + confidence) / 2., n-1)

    return mu, h, mu-h, mu+h


def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False
    otherwise.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
