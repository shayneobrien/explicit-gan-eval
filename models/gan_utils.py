import torch
from torch.autograd import Variable
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import pandas as pd
import numpy as np

from torch.utils.data import TensorDataset
from scipy.stats import entropy, ks_2samp, moment, wasserstein_distance, energy_distance

# Gradient, CUDA
def to_var(x):
    """ Make a tensor cuda-erized and requires gradient """
    return to_cuda(x).requires_grad_()


def to_cuda(x):
    """ Cuda-erize a tensor """
    if torch.cuda.is_available():
        x = x.cuda()
    return x

# Loading data
def get_the_data(generator, samples, BATCH_SIZE=100):
    """ Sample data from respective distribution under consideration,
    make a data loader out of it """
    data = torch.from_numpy(generator.generate_samples(samples)).float()
    labels = torch.from_numpy(np.zeros((samples, 1)))
    data = TensorDataset(data, labels)
    data_iter = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
    return data_iter


def preprocess(generator, samples, BATCH_SIZE=100):
    """ Create data iterators """
    train_iter = get_the_data(generator, samples, BATCH_SIZE)
    val_iter = get_the_data(generator, samples, BATCH_SIZE)
    test_iter = get_the_data(generator, samples, BATCH_SIZE)
    return train_iter, val_iter, test_iter


# Computing metrics for different archtypes
def compute_divergences(A, B):
    """ Compute divergence metrics (Jensen Shannon, Kullback-Liebler,
    Wasserstein Distance, Energy Distance) between predicted distribution A
    and true distribution B """

    # Get number of samples, IQR statistics, range
    samples = A.shape[0]
    iqr = np.percentile(A, 75)-np.percentile(A, 25)
    r = np.max(A) - np.min(A)

    # Get PDFs of predicted distribution A, true distribution B
    A = get_pdf(A, iqr, r, samples)
    B = get_pdf(B, iqr, r, samples)

    # Mean
    m = (np.array(A)+np.array(B))/2

    # Compute metrics
    kl = entropy(pk=A, qk=B).sum()/A.shape[1]
    js = .5*(entropy(pk=A, qk=m)+entropy(pk=B, qk=m)).sum()/A.shape[1]
    wd = sum([wasserstein_distance(A[:,i], B[:,i]) for i in range(A.shape[1])])
    ed = sum([energy_distance(A[:,i], B[:,i]) for i in range(A.shape[1])])

    divergences = {"KL-Divergence": kl,
                    "Jensen-Shannon": js,
                    "Wasserstein-Distance": wd,
                    "Energy-Distance": ed,}

    return divergences


def get_pdf(data, iqr, r, samples):
    """ Compute optimally binned probability distribution function  """
    x = []

    if iqr > 1e-5:
        bin_width = 2*iqr/np.cbrt(samples)
        bins = int(round(r/bin_width, 0))
    else:
        # MNIST (since it's really only supposed to be either 0 or 1 as output)
        bins = 2

    # Bin data
    for i in range(data.shape[1]):
        x.append(list(np.histogram(data[:, i], bins=bins, density=True)[0]))
    res = np.array(x).T
    res[res == 0] = .00001
    return res


def gan_metrics(trainer):
    """ Generate samples, compute divergences, get losses for GANs """

    # Compute divergences for each epoch
    for A, B in zip(*[trainer.As, trainer.Bs]):
        metrics = compute_divergences(A, B)
        for key, value in metrics.items():
            trainer.metrics[key].append(value)

    # Put all metrics into a single dictionary
    metrics = trainer.metrics
    metrics["GLoss"] = trainer.Glosses
    metrics["DLoss"] = trainer.Dlosses
    metrics["LR"] = trainer.lr
    metrics["HDIM"] = trainer.model.hidden_dim
    metrics["BSIZE"] = trainer.train_iter.batch_size

    return metrics


def vae_metrics(trainer):
    """ Generate samples, compute divergences, get losses for VAEs """

    # Compute divergences for each epoch
    for A, B in zip(*[trainer.As, trainer.Bs]):
        metrics = compute_divergences(A, B)
        for key, value in metrics.items():
            trainer.metrics[key].append(value)

    # Put all metrics into a single dictionary
    metrics = trainer.metrics
    metrics["RLoss"] = trainer.Rlosses
    metrics["KL-Divergence"] = trainer.KLdivs
    metrics["LR"] = trainer.lr
    metrics["HDIM"] = trainer.model.hidden_dim
    metrics["BSIZE"] = trainer.train_iter.batch_size

    return metrics


def autoencoder_metrics(trainer):
    """ Generate samples, compute divergences, get losses for autoencoders """

    # Compute divergences for each epoch
    for A, B in zip(*[trainer.As, trainer.Bs]):
        metrics = compute_divergences(A, B)
        for key, value in metrics.items():
            trainer.metrics[key].append(value)

    metrics = trainer.metrics
    metrics["Loss"] = trainer.losses
    metrics["LR"] = trainer.lr
    metrics["HDIM"] = trainer.model.hidden_dim
    metrics["BSIZE"] = trainer.train_iter.batch_size

    return metrics


def sample_gan(trainer):
    """ Sample GAN for metric divergence computation """

    # Set to eval mode (disable regularization)
    trainer.model.eval()

    # TODO: figure out iterating over train iter or not..

    # Sample noise
    noise = trainer.compute_noise(1000, trainer.model.z_dim)

    # Change image shape, if applicable
    A = trainer.process_batch(trainer.train_iter).cpu().data.numpy()

    # Generate from noise
    B = trainer.model.G(noise).cpu().data.numpy()

    return A, B

def sample_autoencoder(output, batch):
    """ Sample GAN for metric divergence computation """

    # Set to eval mode (disable regularization)
    trainer.model.eval()

    # Extract images
    images, _ = batch

    # Sent to numpy
    A = output.cpu().data.numpy()
    B = images.cpu().data.numpy()

    # MNIST, circles make sure shapes are the same
    if A.shape != B.shape:
        B = np.reshape(B, A.shape)

    return A, B
