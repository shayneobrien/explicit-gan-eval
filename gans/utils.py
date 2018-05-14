import torch
from torch.autograd import Variable
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset


def to_var(x):
    """ function to automatically cudarize.. """
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def get_pdf(data):
    x = []
    for i in range(data.shape[1]):
        x.append(list(np.histogram(data[:, i], bins=100, density=True)[0]))
    df = pd.DataFrame(x)
    pdf = list(df.mean(axis=0))
    return pdf


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
