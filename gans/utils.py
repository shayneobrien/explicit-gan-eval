import torch
from torch.autograd import Variable
import pandas as pd
import numpy as np


def to_var(x):
    """ function to automatically cudarize.. """
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def get_pdf(data):
    x = []
    for i in range(data.shape[0]):
        x.append(list(np.histogram(data[i], bins=100, density=True)[0]))
    df = pd.DataFrame(x)
    pdf = list(df.mean(axis=0))
    return pdf
