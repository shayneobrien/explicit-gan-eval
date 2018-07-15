import torch
from torch.autograd import Variable
import pandas as pd
import numpy as np

from torch.utils.data import TensorDataset
from scipy.stats import entropy, ks_2samp, moment, wasserstein_distance, energy_distance


def to_var(x):
    """ Make a tensor cuda-erized and requires gradient """
    return to_cuda(x).requires_grad_()


def to_cuda(x):
    """ Cuda-erize a tensor """
    if torch.cuda.is_available():
        x = x.cuda()
    return x


def get_pdf(data, iqr, r, samples):
    """ Compute optimally binned probability distribution function  """
    x = []
    bin_width = 2*iqr/np.cbrt(samples)
    bins = int(round(r/bin_width,0))
    for i in range(data.shape[1]):
        x.append(list(np.histogram(data[:, i], bins=bins, density=True)[0]))
    res = np.array(x).T
    res[res == 0] = .00001
    return res


def get_the_data(generator, samples, BATCH_SIZE=100):
    """ Sample data from respective distribution under consideration,
    make a data loader out of it """
    data = torch.from_numpy(generator.generate_samples(samples)).float()
    labels = torch.from_numpy(np.zeros((samples, 1)))
    data = TensorDataset(data, labels)
    data_iter = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
    return data_iter


def preprocess(generator, samples, BATCH_SIZE=100):
    """ Create data iterators """
    train_iter = get_the_data(generator, samples, BATCH_SIZE)
    val_iter = get_the_data(generator, samples, BATCH_SIZE)
    test_iter = get_the_data(generator, samples, BATCH_SIZE)
    return train_iter, val_iter, test_iter


def compute_divergences(A, B):
    """ Compute divergence metrics (Jensen Shannon, Kullback-Liebler,
    Wasserstein Distance, Energy Distance) between predicted distribution A
    and true distribution B """

    # Get number of samples, IQR statistics, range
    samples = A.shape[0]
    iqr = np.percentile(A, 75)-np.percentile(A, 25)
    r = np.max(A) - np.min(A)

    # Get PDFs of predicted distribution A, true distribution B
    A = get_pdf(A, iqr, r, samples)
    B = get_pdf(B, iqr, r, samples)

    # TODO: Matt what is this variable?
    m = (np.array(A)+np.array(B))/2

    # Compute metrics
    kl = entropy(pk=A, qk=B).sum()/A.shape[1]
    js = .5*(entropy(pk=A, qk=m)+entropy(pk=B, qk=m)).sum()/A.shape[1]
    wd = sum([wasserstein_distance(A[:,i], B[:,i]) for i in range(A.shape[1])])
    ed = sum([energy_distance(A[:,i], B[:,i]) for i in range(A.shape[1])])

    divergences = {"KL-Divergence": kl,
                    "Jensen-Shannon": js,
                    "Wasserstein-Distance": wd,
                    "Energy-Distance": ed,}

    return divergences

def get_metrics(trainer):
    """ Generate samples, compute divergences, get losses """
    noise = trainer.compute_noise(1000, trainer.model.z_dim)
    a = trainer.process_batch(trainer.train_iter)
    b = trainer.model.G(noise)
    a = a.data.numpy()
    b = b.data.numpy()
    metrics = compute_divergences(a, b)
    metrics['GLoss'] = trainer.Glosses
    metrics['DLoss'] = trainer.Dlosses

    return metrics
