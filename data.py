""" Multivariate distribution data generation file.

We also include methods for generating synthetic multivariate, mixture and
colored circles datasets--these are potential areas of future work. """

try:
   import cPickle as pickle
except:
   import pickle

import os
import scipy.stats as st
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from sklearn.datasets import make_spd_matrix


def load_dist(self, in_file):
    with open(in_file, 'wb') as of:
        return pickle.load(of)


class Distribution:

    def __init__(self, dist_type='normal', dim=1):

        self.dist_type = dist_type
        self.dim = dim
        self.params = {}

        if self.dist_type == 'normal':

            cov_matrix = make_spd_matrix(self.dim)
            mean_matrix = np.random.rand(dim, 1)

            self.params['mean'] = np.squeeze(mean_matrix)
            self.params['covariance'] = cov_matrix

        elif self.dist_type == 'beta':

            rand_a = np.random.rand(dim, 1)
            rand_b = np.random.rand(dim, 1)

            self.params['a'] = np.squeeze(rand_a)
            self.params['b'] = np.squeeze(rand_b)

        elif self.dist_type == 'exponential':

            beta_dist = np.random.rand(dim, 1)

            self.params['beta'] = np.squeeze(beta_dist)

        elif self.dist_type == 'gamma':

            k = np.random.rand(dim, 1) * 10
            theta = np.random.rand(dim, 1) * 2

            self.params['k'] = np.squeeze(k)
            self.params['theta'] = np.squeeze(theta)

        elif self.dist_type == 'gumbel' or self.dist_type == 'laplace':

            loc = np.random.rand(dim, 1)
            scale = np.random.rand(dim, 1)

            self.params['loc'] = np.squeeze(loc)
            self.params['scale'] = np.squeeze(scale)


    def generate_samples(self, n_samples=10000):

        if self.dist_type == 'normal':

            return np.random.multivariate_normal(self.params['mean'], self.params['covariance'], n_samples)

        else:

            samples = []

            for i in range(n_samples):

                if self.dist_type == 'beta':

                    samples.append(np.expand_dims(np.random.beta(self.params['a'], self.params['b']), axis=0))

                elif self.dist_type == 'exponential':

                    samples.append(np.expand_dims(np.random.exponential(self.params['beta']), axis=0))

                elif self.dist_type == 'gamma':

                    samples.append(np.expand_dims(np.random.gamma(self.params['k'], self.params['theta']), axis=0))

                elif self.dist_type == 'gumbel':

                    samples.append(np.expand_dims(np.random.gumbel(self.params['loc'], self.params['scale']), axis=0))

                elif self.dist_type == 'laplace':

                    samples.append(np.expand_dims(np.random.laplace(self.params['loc'], self.params['scale']), axis=0))

                else:

                    raise AttributeError('Invalid distribution type.')

            return np.concatenate(samples, axis=0)

    def generate_high_dimensional_samples(self, n_samples=10000, big_dim=100):
        m = st.ortho_group.rvs(big_dim=dim)
        m_transform = m[:self.dim, :]

        samples = self.generate_samples(n_samples)

        return np.dot(samples, m_transform)


    def save_dist(self, out_file):

        with open(out_file, 'wb') as of:
            pickle.dump(self, of, pickle.HIGHEST_PROTOCOL)


    def get_log_likelihood(self, samples):

        log_likelihood = 0.0
        for sample in samples:

            if self.dist_type == 'normal':

                log_likelihood += st.multivariate_normal.logpdf(sample, mean = self.params['mean'], cov = self.params['covariance'])

            else:

                sample_log_likelihood = 0.0
                for j, dim in enumerate(sample):

                    if self.dist_type == 'beta':

                        sample_log_likelihood += st.beta.logpdf(sample[j], a = self.params['a'][j], b = self.params['b'][j])

                    elif self.dist_type == 'exponential':

                        sample_log_likelihood += st.expon.logpdf(sample[j], scale = self.params['beta'][j])

                    elif self.dist_type == 'gamma':

                        sample_log_likelihood += st.gamma.logpdf(sample[j], a=self.params['k'][j], scale=self.params['theta'][j])

                    elif self.dist_type == 'gumbel':

                        sample_log_likelihood += st.gumbel_r.logpdf(sample[j], loc = self.params['loc'][j], scale = self.params['scale'][j])

                    elif self.dist_type == 'laplace':

                        sample_log_likelihood += st.laplace.logpdf(sample[j], loc = self.params['loc'][j], scale = self.params['scale'][j])

                log_likelihood += sample_log_likelihood

        return log_likelihood
