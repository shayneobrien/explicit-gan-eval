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

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
from scipy.stats import entropy, ks_2samp, moment, wasserstein_distance, energy_distance
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
from itertools import product
from .utils import to_var, get_pdf, get_metrics


class Encoder(nn.Module):
    def __init__(self, image_size, hidden_dim, z_dim):
        """ MLP encoder for VAE. Input is an image, outputs is the mean and std of the latent representation z pre-reparametrization """
        super(Encoder, self).__init__()
        self.linear = nn.Linear(image_size, hidden_dim)
        self.mu = nn.Linear(hidden_dim, z_dim) 
        self.log_var = nn.Linear(hidden_dim, z_dim)
        
    def forward(self, x):
        activated = F.relu(self.linear(x)) #leaky relu?
        mu, log_var = self.mu(activated), self.log_var(activated)
        return mu, log_var


class Decoder(nn.Module):
    """ MLP decoder for VAE. Input is a reparametrized latent representation, output is reconstructed image """
    def __init__(self, z_dim, hidden_dim, image_size):
        super(Decoder, self).__init__()
        self.linear = nn.Linear(z_dim, hidden_dim)
        self.recon = nn.Linear(hidden_dim, image_size)
        
    def forward(self, z):
        activated = F.relu(self.linear(z))
        reconstructed = F.sigmoid(self.recon(activated))
        return reconstructed


class VAE(nn.Module):
    def __init__(self, image_size=784, hidden_dim=400, z_dim=200):
        """ VAE super class to reconstruct an image. Contains reparametrization method """
        super(VAE, self).__init__()
        self.encoder = Encoder(image_size = image_size, hidden_dim = hidden_dim, z_dim = z_dim)
        self.decoder = Decoder(z_dim = z_dim, hidden_dim = hidden_dim, image_size = image_size)
        self.z_dim = z_dim
                     
    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        out_img = self.decoder(z)
        return out_img, mu, log_var
    
    def reparameterize(self, mu, log_var):
        """" Reparametrization trick: z = mean + sigma*epsilon, where epsilon ~ N(0, 1)."""
        epsilon = to_var(torch.randn(mu.size(0), mu.size(1)))
        z = mu + epsilon * torch.exp(log_var/2)    # 2 for convert var to std
        return z


class Trainer:
    def __init__(self, train_iter, val_iter, test_iter, image_data=False):
        """ Object to hold data iterators, train the model """
        self.train_iter = train_iter
        self.val_iter = val_iter
        self.test_iter = test_iter
        self.image_data = image_data
        self.debugging_image, _ = next(iter(val_iter))
        self.kl = []
        self.js = []
        self.ks = []
        self.wd = []
        self.ed = []
    
    def train(self, model, num_epochs, lr = 1e-6, weight_decay = 1e-5):
        """ Train a Variational Autoencoder
            Logs progress using total loss, reconstruction loss, kl_divergence, and validation loss

        Inputs:
            model: class, initialized VAE module
            num_epochs: int, number of epochs to train for
            lr: float, learning rate for Adam optimizer (default 1e-3)
            weight_decay: float, weight decay for Adam optimizer (default 1e-5)
        Outputs:
            model: trained VAE instance """   
        
        # Initialize best validation loss for early stopping
        best_val_loss = 1e10
        
        # Adam optimizer, sigmoid cross entropy for reconstructing binary MNIST
        optimizer = torch.optim.Adam(params=[p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
        criterion = nn.BCELoss(size_average=False)
        
        # Begin training
        for epoch in tqdm_notebook(range(1, num_epochs + 1)):
            
            model.train()
            epoch_loss, epoch_recon, epoch_kl = [], [], []
            
            for batch in self.train_iter:

                # Zero out gradients
                optimizer.zero_grad()
                # images = self.process_batch(self.train_iter)
                # Compute reconstruction loss, Kullback-Leibler divergence for a batch
                recon_loss, kl_diverge = self.compute_batch(batch, criterion, model)
                batch_loss = recon_loss + kl_diverge # ELBO
                
                # Update parameters
                batch_loss.backward()
                optimizer.step()
                
                # Log metrics
                epoch_loss.append(batch_loss.data[0]), epoch_recon.append(recon_loss.data[0]), epoch_kl.append(kl_diverge.data[0])
            
            # Test the model on the validation set
            model.eval()
            val_loss = self.evaluate(self.val_iter, criterion, model)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_model = model
                best_val_loss = val_loss

            # noise = to_var(torch.randn(images.shape[0], 10*model.z_dim))
            # a = np.array(self.train_iter.dataset.data_tensor)
            # print(model.encoder(noise)[0].data)
            # b = model.encoder(noise)[0].data.numpy()
            # b, _, _ = model(to_var(torch.FloatTensor(a)))
            # b = b.data.numpy()
            # b[b == 0] = .00001
            # a[a == 0] = .00001
            # print("a")
            # a = a/np.max(a)
            # print(np.max(a))

            # print(np.min(a))
            # print("b")
            # print(np.max(b))
            # print(np.min(b))
            # kl, js, wd, ed = get_metrics(a, b, 100)
            # self.kl.append(kl)
            # self.wd.append(wd)
            # self.js.append(js)
            # self.ed.append(ed)   
            # Progress logging
            print ("Epoch[%d/%d], Total Loss: %.4f, Reconst Loss: %.4f, KL Div: %.7f, Val Loss: %.4f" 
                   %(epoch, num_epochs, np.mean(epoch_loss), np.mean(epoch_recon), np.mean(epoch_kl), val_loss))
            
            # Debugging and visualization purposes
            if self.image_data is True:
                fig = self.reconstruct_images(self.debugging_image, epoch, model)
                plt.show()
            
        return best_model, self.kl, self.ks, self.js, self.wd, self.ed
    
    def compute_batch(self, batch, criterion, model):
        """ Compute loss for a batch of examples """
        images, _ = batch
        images = to_var(images.view(images.shape[0], -1))
        # The output is NAN'ing after the 3rd epoch... this is our issue!
        output, mu, log_var = model(images)

        print(output, images)

        recon_loss = criterion(output, images)
        kl_diverge = self.kl_divergence(mu, log_var)
        
        return recon_loss, kl_diverge

    def process_batch(self, iterator):
        """ Generate a process batch to be input into the discriminator D """
        images, _ = next(iter(iterator))
        images = to_var(images.view(images.shape[0], -1))
        return images
    
    def evaluate(self, iterator, criterion, model):
        """ Evaluate on a given dataset """
        loss = []
        for batch in iterator:
            recon_loss, kl_diverge = self.compute_batch(batch, criterion, model)
            batch_loss = recon_loss + kl_diverge
            loss.append(batch_loss.data[0])
            
        loss = np.mean(loss)
        return loss
    
    def reconstruct_images(self, images, epoch, model, save = True):
        """Reconstruct a fixed input at each epoch for progress visualization """
        images = to_var(images.view(images.shape[0], -1))
        reconst_images, _, _ = model(images)
        reconst_images = reconst_images.view(reconst_images.shape[0], 28, 28)
        
        size_figure_grid = int(reconst_images.shape[0]**0.5)
        fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
        for i, j in product(range(size_figure_grid), range(size_figure_grid)):
            ax[i,j].get_xaxis().set_visible(False)
            ax[i,j].get_yaxis().set_visible(False)
            ax[i,j].cla()
            ax[i,j].imshow(reconst_images[i+j].data.numpy(), cmap='gray')
                        
        if save:
            if not os.path.exists('../viz/vae-viz/'):
                os.makedirs('../viz/vae-viz/')
            torchvision.utils.save_image(images.data.cpu(), '../viz/vae-viz/real.png', nrow=size_figure_grid)
            torchvision.utils.save_image(reconst_images.unsqueeze(1).data.cpu(), '../viz/vae-viz/reconst_%d.png' %(epoch), nrow=size_figure_grid)
            
        return fig

    def kl_divergence(self, mu, log_var):
        """ Compute Kullback-Leibler divergence """
        return torch.sum(0.5 * (mu**2 + torch.exp(log_var) - log_var - 1))
    
    def save_model(self, model, savepath):
        """ Save model state dictionary """
        torch.save(model.state_dict(), savepath + 'saved_vae.pth')
    
    def load_model(self, loadpath,  model = None):
        """ Load state dictionary into model. If model not specified, instantiate it """
        if not model:
            model = VAE()
        state = torch.load(loadpath)
        model.load_state_dict(state)
        return model


class Viz:
    def __init__(self, train_iter, val_iter, test_iter, model=None):
        self.model = model
        self.train_iter = train_iter
        self.val_iter = val_iter
        self.test_iter = test_iter

    def sample_images(self, model, num_images = 64):
        """ Viz method 1: generate images by sampling z ~ p(z), x ~ p(x|z,θ) """
        sample = to_var(torch.randn(num_images, model.decoder.linear.in_features))
        sample = model.decoder(sample)
        to_img = ToPILImage()
        img = to_img(make_grid(sample.data.view(num_images, 1, 28, 28)))
        display(img)

    def sample_interpolated_images(self, model):
        """ Viz method 2: sample two random latent vectors from p(z), 
            then sample from their interpolated values"""
        z1 = torch.normal(torch.zeros(model.decoder.linear.in_features), 1)
        z2 = torch.normal(torch.zeros(model.decoder.linear.in_features), 1)
        to_img = ToPILImage()
        for alpha in np.linspace(0, 1, model.decoder.linear.in_features):
            z = to_var(alpha*z1 + (1-alpha)*z2)
            sample = model.decoder(z)
            display(to_img(make_grid(sample.data.view(28, 28).unsqueeze(0))))

    def means_scatterplot(self, num_epochs=10):
        """ Viz method 3: train a VAE with 2 latent variables, compare variational means """
        model = VAE(image_size = 784, hidden_dim = 400, z_dim = 2)
        if torch.cuda.is_available():
            vae.cuda()
        trainer = Trainer(self.train_iter, self.val_iter, self.test_iter)
        model = trainer.train(model, num_epochs)

        data = []
        for batch in self.train_iter:
            images, labels = batch
            images = to_var(images.view(images.shape[0], -1))
            mu, log_var = model.encoder(images)

            for label, (m1, m2) in zip(labels, mu):
                data.append((label, m1.data[0], m2.data[0]))

        df = pd.DataFrame(data, columns=('label', 'm1', 'm2'))
        plt.figure(figsize=(10,10))
        plt.scatter(df['m1'], df['m2'], c=df['label'])
        return model

    def explore_latent_space(self, model):
        """ Viz method 4: explore the latent space representations """
        mu = torch.stack([torch.FloatTensor([m1, m2]) for m1 in np.linspace(-2, 2, 10) for m2 in np.linspace(-2, 2, 10)])
        samples = model.decoder(to_var(mu))
        to_img = ToPILImage()
        display(to_img(make_grid(samples.data.view(-1, 1, 28, 28), nrow=10)))

    def make_all(self):
        """ Execute all viz methods outlined in this class """
        self.sample_images(self.model)
        self.sample_interpolated_images(self.model)
        self.model = self.means_scatterplot(num_epochs=3)
        self.explore_latent_space(self.model)
