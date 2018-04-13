""" Wasserstein GAN as laid out in original paper (WGAN)
https://arxiv.org/abs/1701.07875

WGAN utilizes the Wasserstein distance to produce a value function which 
has better theoretical properties than the vanilla GAN. WGAN requires 
that the discriminator (aka Critic because it is not actually classifying) 
lies in the space of 1-Lipschitz functions, enforced via weight clipping. 
The discriminator approximates the Wasserstein distance.
"""
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm_notebook
from itertools import product
from load_data import get_data

def to_var(x):
    """ function to automatically cudarize.. """
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

# Load in data, separate into data loaders
train_iter, val_iter, test_iter = get_data()

class Generator(nn.Module):
    def __init__(self, image_size, hidden_dim, output_dim):
        """ Generator. Input is noise, output is a generated image. """
        super(Generator, self).__init__()
        self.linear = nn.Linear(image_size, hidden_dim)
        self.generate = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        activated = F.relu(self.linear(x))
        generation = F.sigmoid(self.generate(activated))
        return generation
        
class Discriminator(nn.Module):
    def __init__(self, image_size, hidden_dim, output_dim):
        """ Discriminator / Critic (as it's not trained to classify). Input is an image (real or generated), output is P(generated). """
        super(Discriminator, self).__init__()
        self.linear = nn.Linear(image_size, hidden_dim)
        self.discriminate = nn.Linear(hidden_dim, output_dim)     
        
    def forward(self, x):
        activated = F.relu(self.linear(x))
        discrimination = F.sigmoid(self.discriminate(activated))
        return discrimination
    
class WGAN(nn.Module):
    def __init__(self, image_size, hidden_dim):
        """ Super class to contain both Discriminator (D) and Generator (G) """
        super(WGAN, self).__init__()
        self.G = Generator(image_size, hidden_dim, image_size)
        self.D = Discriminator(image_size, hidden_dim, 1)
    
class Trainer:
    def __init__(self, train_iter, val_iter, test_iter):
        """ Object to hold data iterators, train a GAN variant """
        self.train_iter = train_iter
        self.val_iter = val_iter
        self.test_iter = test_iter
    
    def train(self, model, num_epochs, G_lr = 5e-5, D_lr = 5e-5, D_steps = 5, clip = 0.01):
        """ Train a Wasserstein GAN
            Logs progress using G loss, D loss, G(x), D(G(x)), visualizations of Generator output.

        Inputs:
            model: initialized GAN module instance
            num_epochs: int, number of epochs to train for
            G_lr: learning rate for generator's Adam optimizer (default 5e-5)
            D_lr: learning rate for discriminator's Adam optimizer (default 5e-5)
            D_steps: training step ratio for how often to train D compared to G (default 5)
            clip: bound for parameters [-c, c] to crudely ensure K-Lipschitz (default 0.01, or range [-0.01, 0.01])
        Outputs:
            model: trained WGAN instance """
        G_optimizer = torch.optim.RMSprop(params=[p for p in model.G.parameters() if p.requires_grad], lr=G_lr)
        D_optimizer = torch.optim.RMSprop(params=[p for p in model.D.parameters() if p.requires_grad], lr=D_lr)
        
        # Approximate steps/epoch given D_steps per epoch --> roughly train in the same way as if D_step (1) == G_step (1)
        epoch_steps = int(np.ceil(len(train_iter) / (D_steps))) 
        
        for epoch in tqdm_notebook(range(1, num_epochs + 1)):
            model.train()
            G_losses, G_scores, D_losses, D_scores = [], [], [], []
            
            for _ in range(epoch_steps):
                
                # TRAINING D: Train D for D_steps (original WGAN paper e.g. 5)
                for step in range(D_steps):
                    
                    # Retrieve batch
                    images = self.process_batch(self.train_iter)

                    # Zero out gradients for D
                    D_optimizer.zero_grad()

                    # Train the discriminator using samples from the generator
                    D_score, G_score = self.train_D_step(model, images)
                    D_loss = -(torch.mean(D_score) - torch.mean(G_score)) # E[D(x)] - E[D(G(x'))]
                    
                    # Update parameters
                    D_loss.backward()
                    D_optimizer.step()
                    
                    # Save relevant output for progress logging, clamp weights
                    D_losses.append(D_loss), D_scores.append(D_score), G_scores.append(G_score)
                    
                    # Clamp weights as per original paper (this is a crude way of ensuring K-Lipschitz...)
                    self.clip_D_weights(model, clip)
                                    
                # TRAINING G: Zero out gradients for G. 
                G_optimizer.zero_grad()

                # Sample from the generator, reclassify using discriminator to train generator. Keep same batch as
                # the last step for training D
                G_score = self.train_G_step(model, images)

                # Train the generator using predictions from D on the noise compared to true image labels
                # (learn to generate examples from noise that fool the discriminator)
                G_loss = -(torch.mean(G_score))

                # Update parameters
                G_loss.backward()
                G_optimizer.step()

                # Save relevant output for progress logging
                G_losses.append(G_loss)
                
            # Progress logging
            print ("Epoch[%d/%d], G cost: %.4f, D cost: %.4f"
                   %(epoch, num_epochs, np.mean(G_losses), np.mean(D_losses))) 
            
            # Visualize generator progress
            fig = self.generate_images(model, epoch)
            plt.show()
            
        return model
    
    def train_D_step(self, model, images):
        """ Run 1 step of training for discriminator
            
            G_noise = randomly generated noise, x'
            G_output = G(x'), generated images from noise
            D_score = D(x), probability x is fake where x are true images
            G_score = D(G(x')), probability G(x') is fake where x' is noise
        """      
        
        # Sample from the generator
        G_noise = self.compute_noise(images.shape[0], images.shape[1])
        G_output = model.G(G_noise)
        
        # Score real, generated images
        D_score = model.D(images) # D(x), "real"
        G_score = model.D(G_output) # D(G(x')), "fake"
        return D_score, G_score
    
    def train_G_step(self, model, images):
        """ Run 1 step of training for generator
        
            G_noise = randomly generated noise, x'
            G_output = G(x')
            G_score = D(G(x'))
        """
        G_noise = self.compute_noise(images.shape[0], images.shape[1]) # x'
        G_output = model.G(G_noise) # G(x')
        G_score = model.D(G_output) # D(G(x'))
        return G_score
    
    def compute_noise(self, batch_size, image_size):
        """ Compute random noise for the generator to learn to make images from """
        return to_var(torch.randn(batch_size, image_size))
    
    def generate_images(self, model, epoch, num_outputs = 25, save = True):
        """ Visualize progress of generator learning """
        noise = self.compute_noise(num_outputs, 784)
        images = model.G(noise)
        images = images.view(images.shape[0], 28, 28)
        size_figure_grid = int(num_outputs**0.5)
        fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
        for i, j in product(range(size_figure_grid), range(size_figure_grid)):
            ax[i,j].get_xaxis().set_visible(False)
            ax[i,j].get_yaxis().set_visible(False)
            ax[i,j].cla()
            ax[i,j].imshow(images[i+j].data.numpy(), cmap='gray') 
        
        if save:
            if not os.path.exists('../viz/wasserstein-gan/'):
                os.makedirs('../viz/wasserstein-gan/')
            torchvision.utils.save_image(images.unsqueeze(1).data.cpu(), '../viz/wasserstein-gan/reconst_%d.png' %(epoch), nrow = 5)
        return fig
    
    def process_batch(self, iterator):
        """ Generate a process batch to be input into G or D """
        img, _ = next(iter(iterator))
        images = to_var(img.view(img.shape[0], -1))
        return images
    
    def clip_D_weights(self, model, clip):
        for parameter in model.D.parameters():
            parameter.data.clamp_(-clip, clip)    

    def save_model(self, model, savepath):
        """ Save model state dictionary """
        torch.save(model.state_dict(), savepath + 'saved_gan.pth')
    
    def load_model(self, loadpath,  model = None):
        """ Load state dictionary into model. If model not specified, instantiate it """
        if not model:
            model = WGAN()
        state = torch.load(loadpath)
        model.load_state_dict(state)
        return model

model = WGAN(784, 400)
if torch.cuda.is_available():
    model = model.cuda()
trainer = Trainer(train_iter, val_iter, test_iter)
model = trainer.train(model, 200)

