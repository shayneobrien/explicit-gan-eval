import pandas as pd
import torch
import matplotlib.pyplot as plt
import data
from functools import partial
from collections import defaultdict
from models.gan_utils import preprocess
plt.switch_backend('agg')


def get_multivariate_results(models, distributions, dimensions,
                             epochs, samples, hyperparameters):
    res = defaultdict(partial(defaultdict, defaultdict))
    # lr, dim, bsize = hyperparameters
    for model_name, module in models.items():
        for dist in distributions:
            # TODO: fix Gamma
            if dist == 'gamma':
                continue
            print('\n', model_name, dist)

            gen = data.Distribution(dist_type=dist, dim=dimensions)
            metrics = model_results(module, epochs, hyperparameters,
                                    gen, samples, dimensions)

            for metric, value in metrics.items():
                res[model_name][dist][metric] = value

    return res


def get_mixture_results(models, distributions, dimensions,
                        epochs, samples, n_mixtures):
    res = defaultdict(partial(defaultdict, defaultdict))
    lr, dim, bsize = hyperparameters
    for model_name, module in models.items():
        for dist_i in distributions[0]: # Just normal and other mixture models at the moment
            for dist_j in distributions:
                print(dist_i, dist_j, n_mixtures, dimensions, samples)
                # TODO: mix_type='uniform', or mix_type = 'random' (should there be others?)
                gen = data.MixtureDistribution(dist_type=dist_i, mix_type=dist_j,
                                                n_mixtures=n_mixtures, dim=dimensions)

                metrics = model_results(module, epochs, hyperparameters,
                                        gen, samples, dimensions)

                for metric, value in metrics.items():
                    res[model_name][dist][metric] = value

    return res


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
# def get_mnist_results(gans, epochs):
#     res = {}
#     for key, gan in gans.items():
#         res[key] = {}
#         print(key)
#         print("\n\n\n")
#         res[key]["mnist"] = {}
#         train_iter, val_iter, test_iter = get_data(2000)
#         if key == "vae":
#             continue
#             # model = vae.VAE(image_size=784, hidden_dim=400, z_dim=20)
#             # trainer = vae.Trainer(model, train_iter, val_iter, test_iter)
#             # model, kl, ks, js, wd, ed = trainer.train(model, num_epochs=epochs)
#         else:
#             model = gan.GAN(image_size=784, hidden_dim=256, z_dim=int(round(dimensions/4, 0)))
#             trainer = gan.Trainer(model, train_iter, val_iter, test_iter, mnist=True)
#             metrics = trainer.train(model=model, num_epochs=epochs)
#         res[key]["mnist"]["KL-Divergence"] = kl
#         res[key]["mnist"]["Jensen-Shannon"] = js
#         res[key]["mnist"]["Wasserstein-Distance"] = wd
#         res[key]["mnist"]["Energy-Distance"] = ed
#         res[key]["mnist"]["DLoss"] = dl
#         res[key]["mnist"]["GLoss"] = gl
#     return res


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
