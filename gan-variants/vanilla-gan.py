""" Vanilla GAN using MLP architecture """
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import matplotlib.pyplot as plt
import numpy as np
from itertools import product
from tqdm import tqdm_notebook
from IPython.display import clear_output
from load_data import get_data

def to_var(x):
    """ Utility function to automatically cudarize when converting to Variable """
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

# Load in binarized MNIST data, separate into data loaders
train_iter, val_iter, test_iter = get_data()

class Generator(nn.Module):
    def __init__(self, image_size, hidden_dim, output_dim):
        """ Generator. Input is noise, output is a generated image. """
        super(Generator, self).__init__()
        self.linear = nn.Linear(image_size, hidden_dim)
        self.generate = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        activated = F.relu(self.linear(x))
        generation = F.tanh(self.generate(activated))
        return generation
        
class Discriminator(nn.Module):
    def __init__(self, image_size, hidden_dim, output_dim):
        """ Discriminator / Critic. Input is an image (real or generated), output is P(generated). """
        super(Discriminator, self).__init__()
        self.linear = nn.Linear(image_size, hidden_dim)
        self.discriminate = nn.Linear(hidden_dim, output_dim)     
        
    def forward(self, x):
        activated = F.relu(self.linear(x))
        discrimination = F.sigmoid(self.discriminate(activated))
        return discrimination
    
class GAN(nn.Module):
    def __init__(self, image_size = 784, hidden_dim = 400):
        """ Super class to contain both Discriminator (D) and Generator (G) """
        super(GAN, self).__init__()
        self.G = Generator(image_size, hidden_dim, image_size)
        self.D = Discriminator(image_size, hidden_dim, 1)
    
class Trainer:
    def __init__(self, train_iter, val_iter, test_iter):
        """ Object to hold data iterators, train a vanilla GAN (MLP) """
        self.train_iter = train_iter
        self.val_iter = val_iter
        self.test_iter = test_iter
    
    def train(self, model, num_epochs, G_lr = 2e-4, G_weight_decay = 1e-5, D_lr = 2e-4, D_weight_decay = 1e-5):
        """ Train a vanilla GAN. Logs progress using G loss, D loss, G(x), D(G(x)), visualizations of Generator output.

        Inputs:
            model: initialized GAN module instance
            num_epochs: int, number of epochs to train for
            G_lr: learning rate for generator's Adam optimizer
            G_weight_decay: weight decay for generator's Adam optimizer
            D_lr: learning rate for discriminator's Adam optimizer
            D_weight_decay: weight decay for discriminator's Adam optimizer
        Outputs:
            model: trained GAN instance
        """
        G_optimizer = torch.optim.Adam(params=[p for p in model.G.parameters() if p.requires_grad], lr=G_lr, weight_decay=G_weight_decay)
        D_optimizer = torch.optim.Adam(params=[p for p in model.D.parameters() if p.requires_grad], lr=D_lr, weight_decay=D_weight_decay)
        criterion = nn.BCELoss()
        
        for epoch in tqdm_notebook(range(1, num_epochs + 1)):
            model.train()
            G_losses, G_scores, D_losses, D_scores = [], [], [], []
                
            for (img, _) in self.train_iter: 
                
                # Reshape images
                images = to_var(img.view(img.shape[0], -1))
                
                # TRAINING D: Zero out gradients for D
                D_optimizer.zero_grad()

                # Train the discriminator using samples from the generator, compute loss = loss(D(x)) + loss(D(G(x)))
                D_true_loss, D_discrim_loss, D_score, G_score = self.train_D(model, images, criterion)
                D_loss = D_true_loss + D_discrim_loss
                
                # Log results, backpropagate the discriminator network
                D_losses.append(D_loss), D_scores.append(D_score), G_scores.append(G_score)
                D_loss.backward()
                D_optimizer.step()
                
                # TRAINING G: Zero out gradients for G
                G_optimizer.zero_grad()

                # Train the generator using predictions from D on the noise compared to true image labels
                G_loss, G_noise = self.train_G(model, images, criterion)
                
                # Log results, backpropagate the generator network
                G_losses.append(G_loss)
                G_loss.backward()
                G_optimizer.step()
                
            # Progress logging
            print ("Epoch[%d/%d], G Loss: %.4f, D Loss: %.4f, D(x): %.2f, D(G(x)): %.2f"
                   %(epoch, num_epochs, np.mean(G_losses), np.mean(D_losses), 
                     torch.stack(D_scores).data.mean(), torch.stack(G_scores).data.mean())) 
            
            # Visualize generator progress
            fig = self.generate_images(model, G_noise, epoch)
            display(plt.gcf())
            
        return model
    
    def train_D(self, model, images, criterion):
        """ Run 1 step of training for discriminator
        
            D_output = D(x), probability x is fake where x are true images
            G_noise = randomly compute noise, x'
            G_output = G(x')
            D_discrim = D(G(x')), probability G(x') is fake where x' is noise
        """    
        # Generate labels for the real batch images (all 1, since they are real)
        images_labels = to_var(torch.ones(images.shape[0])) 
        
        # Classify the real batch images, get the loss for these 
        D_output = model.D(images)
        D_true_loss = criterion(D_output.squeeze(), images_labels)
        
        # Sample outputs from the generator
        G_noise = self.compute_noise(images.shape[0], images.shape[1])
        G_output = model.G(G_noise)
        G_labels = to_var(torch.zeros(images.shape[0]))
        
        # Classify the fake batch images, get the loss for these (labels being all 0, since they are fake)
        D_discrim = model.D(G_output)
        D_discrim_loss = criterion(D_discrim.squeeze(), G_labels)
        
        return D_true_loss, D_discrim_loss, D_output, D_discrim
    
    def train_G(self, model, images, criterion):
        """ Run 1 step of training for generator
        
            G_noise = randomly generated noise, x'
            G_output = G(x')
            G_score = D(G(x'))
        """        
        # Generate labels for the generator batch images (all 0, since they are fake)
        images_labels = to_var(torch.ones(images.shape[0])) 
        
        # Sample from the generator, reclassify using discriminator to train generator
        G_noise = self.compute_noise(images.shape[0], images.shape[1])
        G_output = model.G(G_noise)
        D_output = model.D(G_output)
        
        # Compute the loss for how D did versus the generations of G
        G_loss = criterion(D_output.squeeze(), images_labels)
        return G_loss, G_noise
    
    def compute_noise(self, batch_size, image_size):
        """ Compute random noise for the generator to learn to make images from """
        return to_var(torch.randn(batch_size, image_size))
    
    def generate_images(self, model, noise, epoch, num_outputs = 16, save = True):
        """ Visualize progress of generator learning """
        images = model.G(noise)
        size_figure_grid = int(num_outputs**0.5)
        fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
        for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
            ax[i,j].get_xaxis().set_visible(False)
            ax[i,j].get_yaxis().set_visible(False)
            ax[i,j].cla()
            ax[i,j].imshow(images[i+j,:].data.cpu().numpy().reshape(28, 28), cmap='Greys') 
            
        if save:
            plt.savefig('../viz/gan-viz/reconst_%d.png' %(epoch))
        return fig
    
    def save_model(self, model, savepath):
        """ Save model state dictionary """
        torch.save(model.state_dict(), savepath)
    
    def load_model(self, loadpath,  model = None):
        """ Load state dictionary into model. If model not specified, instantiate it """
        if not model:
            model = GAN()
        state = torch.load(loadpath)
        model.load_state_dict(state)
        return model


model = GAN(784, 400)
if torch.cuda.is_available():
    model = model.cuda()
trainer = Trainer(train_iter, val_iter, test_iter)
model = trainer.train(model, 200)
