import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.models.inception import inception_v3

import pandas as pd
import numpy as np
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
def get_the_data(generator, samples, batch_size):
    """ Sample data from respective distribution under consideration,
    make a data loader out of it """
    data = to_cuda(torch.from_numpy(generator.generate_samples(samples)).float())
    labels = to_cuda(torch.from_numpy(np.zeros((samples, 1))))
    data = TensorDataset(data, labels)
    data_iter = DataLoader(data, batch_size=batch_size, shuffle=True)
    return data_iter


def preprocess(generator, samples, batch_size, epochs):
    """ Create data iterators, minimizing the amount of data that needs to be
    generated """
    train_iter = get_the_data(generator, samples, batch_size)
    test_iter = get_the_data(generator, int(batch_size*epochs), batch_size)
    return train_iter, test_iter


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
    B = get_pdf(B, iqr, r, samples)
    A = get_pdf(A, iqr, r, samples)

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
                    "Energy-Distance": ed}

    return divergences


def get_pdf(data, iqr, r, samples):
    """ Compute optimally binned probability distribution function  """
    x = []

    # NOTE: this would be problematic for negative data (none of our datasets are)
    if iqr > 1e-5:
        bin_width = 2*iqr/np.cbrt(samples)
        bins = int(round(r/bin_width, 0))
    else:
        # MNIST (since it's really only supposed to be either 0 or 1 as output)
        # TODO: bin number
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

    # Sample noise
    noise = trainer.compute_noise(trainer.test_iter.batch_size, trainer.model.z_dim)

    # Change image shape, if applicable
    A = trainer.process_batch(trainer.test_iter).cpu().data.numpy()

    # Generate from noise
    B = trainer.model.G(noise).cpu().data.numpy()

    return A, B

def sample_autoencoder(trainer):
    """ Sample autoencoder for metric divergence computation """

    # Set to eval mode (disable regularization)
    trainer.model.eval()

    # If it's not standard autoencoder, sample latent space.
    if 'reparameterize' in dir(trainer.model):

        # Sample z
        z = to_cuda(torch.randn(trainer.train_iter.batch_size,
                                trainer.model.decoder.linear.in_features))

        # Pass into decoder
        B = trainer.model.decoder(z)

        # Extract images
        A, _, _, _ = trainer.process_batch(next(iter(trainer.test_iter)))

    # Otherwise, autoencode.
    else:

        A, B, _, _ = trainer.process_batch(next(iter(trainer.test_iter)))

    # Send to numpy
    A = A.cpu().data.numpy()
    B = B.cpu().data.numpy()

    # MNIST, circles make sure shapes are the same
    if len(A.shape) != len(B.shape):
        print(A.shape, B.shape)
        B = np.reshape(B, A.shape)

    return A, B


def inception_score(images, batch_size=32, resize=False, splits=1):
    """ Credit to: https://github.com/sbarratt/inception-score-pytorch
    (with slight edits for PyTorch 0.4.0+)

    Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N >= batch_size

    # Set up dataloader
    dataloader = DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    # TODO: this can be done faster, takes about 2s
    inception_model = to_cuda(inception_v3(pretrained=True, transform_input=False))
    inception_model.eval()

    def get_pred(x):
        if resize:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        x = inception_model(x)
        return F.softmax(x, dim=1).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = to_cuda(batch)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

def frechet_inception_distance():
    """ https://github.com/mseitzer/pytorch-fid/blob/master/fid_score.py """
    pass
