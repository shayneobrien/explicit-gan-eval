""" (Autoencoder)
Standard Autoencoder

Autoencoders take an input representation, encode it into a reduced dimensionality
space using an 'encoder network', and then decode it using a 'decoder network'
back to its original representation
"""

import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid

import os
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from collections import defaultdict
from tqdm import tqdm
from itertools import product
from .model_utils import to_var, autoencoder_metrics


def to_cuda(x):
    """ Cuda-erize a tensor """
    if torch.cuda.is_available():
        x = x.cuda()
    return x


class Encoder(nn.Module):
    """ Feedforward network encoder. Input is an image, output is encoded
    vector representation of that image.
    """
    def __init__(self, image_size, hidden_dim, atype):
        super(Encoder, self).__init__()

        self.__dict__.update(locals())
        self.linear = nn.Linear(image_size, hidden_dim)

    def forward(self, x):
        if self.atype == 'relu':
            return F.relu(self.linear(x))
        elif self.atype == 'sigmoid':
            return torch.sigmoid(self.linear(x))


class Decoder(nn.Module):
    """ Feedforward network decoder. Input is an encoded vector representation,
    output is reconstructed image.
    """
    def __init__(self, hidden_dim, image_size):
        super(Decoder, self).__init__()

        self.linear = nn.Linear(hidden_dim, image_size)

    def forward(self, encoder_output):
        return torch.sigmoid(self.linear(encoder_output))


class Model(nn.Module):
    """ Autoencoder super class to encode then decode an image
    """
    def __init__(self, image_size, hidden_dim, z_dim, atype):
        super().__init__()
        self.__dict__.update(locals())

        self.encoder = Encoder(image_size=image_size, hidden_dim=hidden_dim, atype=atype)
        self.decoder = Decoder(hidden_dim=hidden_dim, image_size=image_size)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class Trainer:
    def __init__(self, model, train_iter, val_iter, test_iter, viz=False):
        """ Object to hold data iterators, train the model """
        self.model = to_cuda(model)
        self.name = model.__class__.__name__

        self.train_iter = train_iter
        self.val_iter = val_iter
        self.test_iter = test_iter

        self.viz = viz
        if self.viz:
            self.debugging_image, _ = next(iter(test_iter))

        self.metrics = defaultdict(list)
        self.losses = []

    def train(self, num_epochs, lr=1e-3, weight_decay=1e-5):
        """ Train a Standard Autoencoder
            Logs progress using total loss, validation loss

        Inputs:
            num_epochs: int, number of epochs to train for
            lr: float, learning rate for Adam optimizer (default 1e-3)
            weight_decay: float, weight decay for Adam optimizer (default 1e-5)
        """

        # Initialize best validation loss for early stopping
        best_val_loss = 1e10

        # Adam optimizer, sigmoid cross entropy for reconstructing binary MNIST
        optimizer = torch.optim.Adam(params=[p for p in self.model.parameters() if p.requires_grad],
                                     lr=lr,
                                     weight_decay=weight_decay)
        self.__dict__.update(locals())

        # Compute number of steps per epoch
        epoch_steps = int(len(self.train_iter))

        # Begin training
        for epoch in tqdm(range(1, num_epochs+1)):

            self.model.train()
            epoch_loss = []

            for _ in range(epoch_steps):

                # Zero out gradients
                optimizer.zero_grad()

                # Compute reconstruction loss for a batch
                output, _, batch_loss = self.compute_batch(batch)

                # Update parameters
                batch_loss.backward()
                optimizer.step()

                # Log metrics
                epoch_loss.append(batch_loss.item())

            # Append losses
            self.losses.extend(epoch_loss)

            # Test the model on the validation set
            val_loss = self.evaluate(self.val_iter)

            # Early stopping
            if val_loss < best_val_loss:
                self.best_model = self.model
                best_val_loss = val_loss

            # Sample for metric divergence computation, save outputs
            A, B = sample_autoencoder(self)
            self.As.append(A), self.Bs.append(B)

            # Re-cuda model
            self.model = to_cuda(self.model)

            # Progress logging
            print ("Epoch[%d/%d], Train Loss: %.4f, Val Loss: %.4f"
                   %(epoch, num_epochs, np.mean(epoch_loss), val_loss))

            # Debugging and visualization purposes
            if self.viz:
                self.reconstruct_images(self.debugging_image, epoch)
                plt.show()

        return autoencoder_metrics(self)

    def process_batch(self, iterator):
        """ Compute loss for a batch of examples """

        images, _ = next(iter(iterator))
        images = to_cuda(images.view(images.shape[0], -1))

        output = self.model(images)

        # Binary cross entropy
        recon_loss = -torch.sum(images*torch.log(output + 1e-8)
                                 + (1-images) * torch.log(1 - output + 1e-8))

        return output, images, recon_loss

    def evaluate(self, iterator):
        """ Evaluate on a given dataset """
        return np.mean([self.compute_batch(batch)[1].item() for batch in iterator])

    def reconstruct_images(self, images, epoch, save=True):
        """Reconstruct a fixed input at each epoch for progress visualization """
        # Reshape images, VAE output
        images = to_cuda(images.view(images.shape[0], -1))
        reconst_images = self.model(images)
        reconst_images = reconst_images.view(reconst_images.shape[0], 28, 28, -1).squeeze()

        # Plot
        plt.close()
        size_figure_grid, k = int(reconst_images.shape[0]**0.5), 0
        fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
        for i, j in product(range(size_figure_grid), range(size_figure_grid)):
            ax[i,j].get_xaxis().set_visible(False)
            ax[i,j].get_yaxis().set_visible(False)
            ax[i,j].imshow(reconst_images[k].data.numpy(), cmap='gray')
            k += 1

        # Save
        if save:
            outname = '../viz/' + self.name + '/'
            if not os.path.exists(outname):
                os.makedirs(outname)
            torchvision.utils.save_image(images.data,
                                         outname + 'real.png',
                                         nrow=size_figure_grid)
            torchvision.utils.save_image(reconst_images.unsqueeze(1).data,
                                         outname + 'reconst_%d.png' %(epoch),
                                         nrow=size_figure_grid)

    def save_model(self, savepath):
        """ Save model state dictionary """
        torch.save(self.best_model.state_dict(), savepath)

    def load_model(self, loadpath):
        """ Load state dictionary into model. If model not specified, instantiate it """
        state = torch.load(loadpath)
        self.model.load_state_dict(state)
