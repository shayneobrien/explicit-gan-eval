import torch
from torch.autograd import Variable
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import pandas as pd
import numpy as np

from torch.utils.data import TensorDataset
from scipy.stats import entropy, ks_2samp, moment, wasserstein_distance, energy_distance
from .ae import Model, Trainer


def get_the_data_mnist(BATCH_SIZE):
    """ Load data for binared MNIST """
    torch.manual_seed(3435)

    """ Download our data """
    train_dataset = datasets.MNIST(root='./data/',
                                train=True,
                                transform=transforms.ToTensor(),
                                download=True)

    test_dataset = datasets.MNIST(root='./data/',
                               train=False,
                               transform=transforms.ToTensor())

    """ Use greyscale values as sampling probabilities to get back to {0,1} """
    train_img = torch.stack([torch.bernoulli(d[0]) for d in train_dataset])
    train_label = torch.LongTensor([d[1] for d in train_dataset])

    test_img = torch.stack([torch.bernoulli(d[0]) for d in test_dataset])
    test_label = torch.LongTensor([d[1] for d in test_dataset])

    """ MNIST has no official train dataset so use last 10000 as validation """
    val_img = train_img[-10000:].clone()
    val_label = train_label[-10000:].clone()

    train_img = train_img[:-10000]
    train_label = train_label[:-10000]

    """ Create data loaders """
    train = torch.utils.data.TensorDataset(train_img, train_label)
    val = torch.utils.data.TensorDataset(val_img, val_label)
    test = torch.utils.data.TensorDataset(test_img, test_label)
    # BATCH_SIZE = 100
    train_iter = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    val_iter = torch.utils.data.DataLoader(val, batch_size=BATCH_SIZE, shuffle=True)
    test_iter = torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE, shuffle=True)
    return train_iter, val_iter, test_iter


def preprocess_mnist(BATCH_SIZE=100):
    """ Create data iterators """
    train_iter, val_iter, test_iter = get_the_data_mnist(BATCH_SIZE)
    # Here the intention is to run the autoencoder on the MNIST data and output the autoencoded data as train_iter, val_iter, test_iter
    model = Model(image_size=784,
                  hidden_dim=32)

    trainer = Trainer(model=model,
                      train_iter=train_iter,
                      val_iter=val_iter,
                      test_iter=test_iter,
                      viz=False)

    trainer.train(num_epochs=1,
                  lr=1e-3,
                  weight_decay=1e-5)

    autoencoder_mnist = {}
    for count, dataset in enumerate([train_iter, val_iter, test_iter]):
        for index, data in enumerate(dataset):
            img, _ = data
            img = img.view(img.size(0), -1)
            img = Variable(img)
            result = trainer.model.forward(img)
            result = result/torch.max(result)
            if index == 0:
                results = result
            else:
                results = torch.cat([results, result])
        autoencoder_mnist[str(count)] = results
    return  autoencoder_mnist["0"], autoencoder_mnist["1"], autoencoder_mnist["2"]
