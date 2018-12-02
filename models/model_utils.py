""" Code for computing divergence metrics
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
from scipy.stats import entropy, wasserstein_distance


def to_var(x):
    """ Make a tensor cuda-erized and requires gradient """
    return to_cuda(x).requires_grad_()


def to_cuda(x):
    """ Cuda-erize a tensor """
    if torch.cuda.is_available():
        x = x.cuda()
    return x

def get_the_data(generator, samples, batch_size):
    """ Sample data from respective distribution under consideration,
    make a data loader out of it """
    data = to_cuda(torch.from_numpy(generator.generate_samples(samples)).float())
    labels = to_cuda(torch.from_numpy(np.zeros((samples, 1))))
    data = TensorDataset(data, labels)
    data_iter = DataLoader(data, batch_size=batch_size, shuffle=True)
    return data_iter


def preprocess(generator, samples, batch_size, epochs):
    """ Create data iterators, minimizing the amount of data that needs to be
    generated """
    train_iter = get_the_data(generator, samples, batch_size)
    test_iter = get_the_data(generator, int(batch_size*epochs), batch_size)
    return train_iter, test_iter


# Computing metrics for different archtypes
def compute_divergences(A, B):
    """ Compute divergence metrics (Jensen Shannon, Kullback-Liebler,
    Wasserstein Distance, Energy Distance) between predicted distribution A
    and true distribution B """

    # Get number of samples, IQR statistics, range
    samples = A.shape[0]
    iqr = np.percentile(A, 75)-np.percentile(A, 25)
    r = np.max(A) - np.min(A)

    # Get PDFs of true distribution A, predicted distribution B
    A = get_pdf(A, iqr, r, samples)
    B = get_pdf(B, iqr, r, samples)

    # Mean
    m = (np.array(A)+np.array(B))/2

    # Compute metrics
    kl = entropy(pk=A, qk=B).sum()/A.shape[1]
    js = .5*(entropy(pk=A, qk=m)+entropy(pk=B, qk=m)).sum()/A.shape[1]
    wd = sum([wasserstein_distance(A[:,i], B[:,i]) for i in range(A.shape[1])])

    divergences = {"KL-Divergence": kl,
                    "Jensen-Shannon": js,
                    "Wasserstein-Distance": wd
                    }

    return divergences


def get_pdf(data, iqr, r, samples):
    """ Compute optimally binned probability distribution function  """
    x = []

    # NOTE: this would be problematic for negative data (none of our datasets are)
    bin_width = 2*iqr/np.cbrt(samples)
    bins = int(round(r/bin_width, 0))
    
    # Bin data
    for i in range(data.shape[1]):
        x.append(list(np.histogram(data[:, i], bins=bins, density=True)[0]))
    res = np.array(x).T
    res[res == 0] = .00001
    return res


def gan_metrics(trainer):
    """ Generate samples, compute divergences, get losses for GANs """

    # Compute divergences for each epoch
    for A, B in zip(*[trainer.As, trainer.Bs]):
        metrics = compute_divergences(A, B)
        for key, value in metrics.items():
            trainer.metrics[key].append(value)

    # Put all metrics into a single dictionary
    metrics = trainer.metrics
    metrics["GLoss"] = trainer.Glosses
    metrics["DLoss"] = trainer.Dlosses
    metrics["LR"] = trainer.lr
    metrics["HDIM"] = trainer.model.hidden_dim
    metrics["BSIZE"] = trainer.train_iter.batch_size

    return metrics

def sample_gan(trainer):
    """ Sample GAN for metric divergence computation """

    # Set to eval mode (disable regularization)
    trainer.model.eval()

    # Sample noise
    noise = trainer.compute_noise(trainer.test_iter.batch_size, trainer.model.z_dim)

    # Change image shape, if applicable
    A = trainer.process_batch(trainer.test_iter).cpu().data.numpy()

    # Generate from noise
    B = trainer.model.G(noise).cpu().data.numpy()

    return A, B
