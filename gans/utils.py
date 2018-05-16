import torch
from torch.autograd import Variable
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset
from scipy.stats import entropy, ks_2samp, moment, wasserstein_distance, energy_distance


def to_var(x):
    """ function to automatically cudarize.. """
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def get_pdf(data, iqr, r, samples):
    x = []
    bin_width = 2*iqr/np.cbrt(samples)
    bins = int(round(r/bin_width,0))
    for i in range(data.shape[1]):
        x.append(list(np.histogram(data[:, i], bins=bins, density=True)[0]))
    res = np.array(x).T
    res[res == 0] = .00001
    return res


def get_the_data(generator, samples, BATCH_SIZE=100):
    data = torch.from_numpy(generator.generate_samples(samples)).float()
    labels = torch.from_numpy(np.zeros((samples, 1)))
    data = TensorDataset(data, labels)
    data_iter = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
    return data_iter


def preprocess(generator, samples, BATCH_SIZE=100):
    train_iter = get_the_data(generator, samples, BATCH_SIZE)
    val_iter = get_the_data(generator, samples, BATCH_SIZE)
    test_iter = get_the_data(generator, samples, BATCH_SIZE)
    return train_iter, val_iter, test_iter


def get_metrics(a, b):
    samples = a.shape[0]
    iqr = np.percentile(a, 75)-np.percentile(a, 25)
    r = np.max(a) - np.min(a)
    a = get_pdf(a, iqr, r, samples)
    b = get_pdf(b, iqr, r, samples)
    m = (np.array(a)+np.array(b))/2
    kl = entropy(pk=a, qk=b).sum()/a.shape[1]
    jshannon = .5*(entropy(pk=a, qk=m)+entropy(pk=b, qk=m)).sum()/a.shape[1]
    wd = 0
    for i in range(a.shape[1]):
        wd += wasserstein_distance(a[:,i],b[:,i])
    ed = 0
    for i in range(a.shape[1]):
        ed += energy_distance(a[:,i],b[:,i])
    return kl, jshannon, wd, ed