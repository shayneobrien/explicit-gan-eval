import torch
from torch.autograd import Variable
import torchvision.datasets as datasets
from torch.utils.data import TensorDataset
import torchvision.transforms as transforms

import os, random
import pandas as pd
import numpy as np
from klepto.archives import file_archive

import models.ae as ae
from models.model_utils import to_cuda


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
    train_img = to_cuda(torch.stack([torch.bernoulli(d[0]) for d in train_dataset]))
    train_label = to_cuda(torch.LongTensor([d[1] for d in train_dataset]))

    test_img = to_cuda(torch.stack([torch.bernoulli(d[0]) for d in test_dataset]))
    test_label = to_cuda(torch.LongTensor([d[1] for d in test_dataset]))

    """ MNIST has no official validation dataset so use last 10000 as validation """
    # TODO: yikes...? but this is consistent with literature
    val_img = train_img[-10000:].clone()
    val_label = train_label[-10000:].clone()

    train_img = train_img[:-10000]
    train_label = train_label[:-10000]

    """ Create data loaders """
    train = torch.utils.data.TensorDataset(train_img, train_label)
    val = torch.utils.data.TensorDataset(val_img, val_label)
    test = torch.utils.data.TensorDataset(test_img, test_label)

    train_iter = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    val_iter = torch.utils.data.DataLoader(val, batch_size=BATCH_SIZE, shuffle=True)
    test_iter = torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE, shuffle=True)
    return train_iter, val_iter, test_iter


def preprocess_mnist(BATCH_SIZE=100, save_path='data/autoencoder', overwrite=False):
    """ Here the intention is to run the autoencoder on the MNIST data
    and output the autoencoded data as train_iter, val_iter, test_iter
    """
    activation_type = 'sigmoid'

    # If model not yet trained train and save it
    if (not os.path.exists(save_path + '/cached_autoencoder.pth')) and (not overwrite):
        try:
            os.makedirs(save_path)
        except FileExistsError:
            pass

        # Load MNIST data
        train_iter, val_iter, test_iter = get_the_data_mnist(BATCH_SIZE)

        # Train autoencoder
        print('Training autoencoder...')
        model = ae.Model(image_size=784,
                         hidden_dim=512,
                         z_dim=0,
                         atype=activation_type)

        trainer = ae.Trainer(model=model,
                              train_iter=train_iter,
                              val_iter=val_iter,
                              test_iter=test_iter,
                              viz=False)

        _ = trainer.train(num_epochs=25,
                          lr=1e-3,
                          weight_decay=1e-5)

        # Cache autoencoder
        trainer.save_model(save_path + '/cached_autoencoder.pth')

    else:

        model = ae.Model(image_size=784,
                         hidden_dim=512,
                         z_dim=0,
                         atype=activation_type)

        trainer = ae.Trainer(model=model,
                              train_iter=None,
                              val_iter=None,
                              test_iter=None,
                              viz=False)

        # Load cached autoencoder
        trainer.load_model(save_path + '/cached_autoencoder.pth')

    # Load cached predictions if they exist, otherwise make them
    if not os.path.exists(save_path + '/cached_preds.txt') and not overwrite:
        print('Initializing autoencoded data...')
        try:
            os.makedirs(save_path)
        except FileExistsError:
            pass

        if 'train_iter' not in locals():
            # Load MNIST data
            train_iter, val_iter, test_iter = get_the_data_mnist(BATCH_SIZE)

        # Init cache
        cache = file_archive(save_path + '/cached_preds.txt')

        # Set to evaluation mode
        trainer.model.eval()
        results = []
        for count, dataset in enumerate([train_iter, val_iter, test_iter]):

            # Autoencode all images
            for batch in dataset:

                images, labels = batch
                images = to_cuda(images.view(images.shape[0], -1))

                output = trainer.model(images)
                results.append((output.detach(), labels.detach()))

            # In place shuffle
            random.shuffle(results)

            # Make a dataset out of the autoencoded images, copy attributes
            autoencoded_data = list_obj(results)
            autoencoded_data.__dict__ = dataset.__dict__.copy()

            # Store into dictionary
            cache[str(count)] = autoencoded_data

            # Reinitialize results for the next dataset
            results = []

        cache.dump()

    else:

        # Load data
        cache = file_archive(save_path + '/cached_preds.txt')
        cache.load()

    return cache["0"], cache["1"], cache["2"]


class list_obj(list):
    """ Used in preprocess_mnist so we can set attributes that are later
    used in metrics functions """
    pass
