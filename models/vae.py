""" (VAE)
Variational Autoencoder

https://arxiv.org/pdf/1312.6114.pdf

From the abstract:

"We introduce a stochastic variational inference and learning algorithm that scales to large datasets
and, under some mild differentiability conditions, even works in the intractable case. Our contributions
is two-fold. First, we show that a reparameterization of the variational lower bound yields a lower bound
estimator that can be straightforwardly optimized using standard stochastic gradient methods. Second, we
show that for i.i.d. datasets with continuous latent variables per datapoint, posterior inference can be
made especially efficient by fitting an approximate inference model (also called a recognition model) to
the intractable posterior using the proposed lower bound estimator."

Basically VAEs encode an input into a given dimension z, reparametrize that z using it's mean and std, and
then reconstruct the image from reparametrized z. This lets us tractably model latent representations that we
may not be explicitly aware of that are in the data. For a simple example of what this may look like, read
up on "Karl Pearson's Crabs." The basic idea was that a scientist collected data on a population of crabs,
noticed that the distribution was non-normal, and Pearson postulated it was because there were likely
more than one population of crabs studied. This would've been a latent variable, since the data colllector
did not know.
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
from itertools import product
from tqdm import tqdm

from .model_utils import *


class Encoder(nn.Module):
    """ MLP encoder for VAE. Input is an image,
    outputs is the mean and std of the latent representation z pre-reparametrization
    """
    def __init__(self, image_size, hidden_dim, z_dim):
        super().__init__()

        self.linear = nn.Linear(image_size, hidden_dim)
        self.mu = nn.Linear(hidden_dim, z_dim)
        self.log_var = nn.Linear(hidden_dim, z_dim)

    def forward(self, x):
        activated = F.relu(self.linear(x)) #leaky relu?
        mu, log_var = self.mu(activated), self.log_var(activated)
        return mu, log_var


class Decoder(nn.Module):
    """ MLP decoder for VAE. Input is a reparametrized latent representation,
    output is reconstructed image """
    def __init__(self, z_dim, hidden_dim, image_size):
        super().__init__()

        self.linear = nn.Linear(z_dim, hidden_dim)
        self.recon = nn.Linear(hidden_dim, image_size)

    def forward(self, z):
        activated = F.relu(self.linear(z))
        reconstructed = torch.sigmoid(self.recon(activated))
        return reconstructed


class Model(nn.Module):
    """ VAE super class to reconstruct an image. Contains reparametrization method
    """
    def __init__(self, image_size, hidden_dim, z_dim, atype):
        super().__init__()
        self.__dict__.update(locals())

        self.encoder = Encoder(image_size=image_size, hidden_dim=hidden_dim, z_dim=z_dim)
        self.decoder = Decoder(z_dim=z_dim, hidden_dim=hidden_dim, image_size=image_size)

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        out_img = self.decoder(z)
        return out_img, mu, log_var

    def reparameterize(self, mu, log_var):
        """" Reparametrization trick: z = mean + std*epsilon, where epsilon ~ N(0, 1)."""
        epsilon = to_cuda(torch.randn(mu.size(0), mu.size(1)))
        z = mu + epsilon * torch.exp(log_var/2)    # 2 for convert var to std
        return z


class Trainer:
    def __init__(self, model, train_iter, val_iter, test_iter, viz=False):
        """ Object to hold data iterators, train the model """
        self.model = to_cuda(model)
        self.name = model.__class__.__name__

        self.train_iter = train_iter
        self.val_iter = val_iter
        self.test_iter = test_iter

        self.debugging_image, _ = next(iter(test_iter))
        self.viz = viz
        self.metrics = defaultdict(list)

        self.Rlosses = []
        self.KLdivs = []

        self.As = []
        self.Bs = []

    def train(self, num_epochs, lr=1e-3, weight_decay=1e-5):
        """ Train a Variational Autoencoder
            Logs progress using total loss, reconstruction loss, kl_divergence, and validation loss

        Inputs:
            num_epochs: int, number of epochs to train for
            lr: float, learning rate for Adam optimizer (default 1e-3)
            weight_decay: float, weight decay for Adam optimizer (default 1e-5)
        """

        # Initialize best validation loss for early stopping
        best_val_loss = 1e10

        # Adam optimizer, sigmoid cross entropy for reconstructing binary MNIST
        optimizer = torch.optim.Adam(params=[p for p in self.model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
        self.__dict__.update(locals())

        # Begin training
        for epoch in tqdm(range(1, num_epochs+1)):

            self.model.train()
            epoch_loss, epoch_recon, epoch_kl = [], [], []

            for batch in self.train_iter:

                # Zero out gradients
                optimizer.zero_grad()

                # Compute reconstruction loss, Kullback-Leibler divergence for a batch
                _, _, recon_loss, kl_diverge = self.process_batch(batch)
                batch_loss = recon_loss + kl_diverge # ELBO

                # Update parameters
                batch_loss.backward()
                optimizer.step()

                # Log metrics
                epoch_loss.append(batch_loss.item())
                epoch_recon.append(recon_loss.item())
                epoch_kl.append(kl_diverge.item())

            # Save progress
            self.Rlosses.extend(epoch_recon)
            self.KLdivs.extend(epoch_kl)

            # # Test the model on the validation set
            # val_loss = self.evaluate(self.val_iter)

            # # Early stopping
            # if val_loss < best_val_loss:
            #     self.best_model = self.model
            #     best_val_loss = val_loss

            # Sample for metric divergence computation, save outputs
            A, B = sample_autoencoder(self)
            self.As.append(A), self.Bs.append(B)

            # Re-cuda model
            self.model = to_cuda(self.model)

            # Progress logging
            print ("Epoch[%d/%d], Total Loss: %.4f, Reconst Loss: %.4f, KL Div: %.7f"
                   %(epoch, num_epochs, np.mean(epoch_loss),
                    np.mean(epoch_recon), np.mean(epoch_kl)))

            # Debugging and visualization purposes
            if self.viz:
                self.reconstruct_images(self.debugging_image, epoch)
                plt.show()

        return vae_metrics(self)

    def process_batch(self, batch):
        """ Compute loss for a batch of examples """

        # Reshape
        images, _ = batch
        images = to_cuda(images.view(images.shape[0], -1))

        # Get output images, mean, std of encoded space
        outputs, mu, log_var = self.model(images)

        # L2 (mean squared error) loss
        recon_loss = torch.sum((images - outputs) ** 2)

        # Kullback-Leibler divergence between encoded space, Gaussian
        kl_diverge = self.kl_divergence(mu, log_var)

        return outputs, images, recon_loss, kl_diverge

    def evaluate(self, iterator):
        """ Evaluate on a given dataset """
        loss = []
        for batch in iterator:
            output, _, recon_loss, kl_diverge = self.process_batch(batch)
            batch_loss = recon_loss + kl_diverge
            loss.append(batch_loss.item())

        loss = np.mean(loss)
        return loss

    def reconstruct_images(self, images, epoch, save=True):
        """Reconstruct a fixed input at each epoch for progress visualization """
        images = to_cuda(images.view(images.shape[0], -1))
        reconst_images, _, _ = self.model(images)
        reconst_images = reconst_images.view(reconst_images.shape[0], 28, 28, -1).squeeze()

        # Plot
        plt.close()
        size_figure_grid, k = int(reconst_images**0.5), 0
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

    def kl_divergence(self, mu, log_var):
        """ Compute Kullback-Leibler divergence """
        return torch.sum(0.5 * (mu**2 + torch.exp(log_var) - log_var - 1))

    def save_model(self, savepath):
        """ Save model state dictionary """
        torch.save(self.best_model.state_dict(), savepath)

    def load_model(self, loadpath):
        """ Load state dictionary into model. If model not specified, instantiate it """
        state = torch.load(loadpath)
        self.model.load_state_dict(state)
