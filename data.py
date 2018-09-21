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


class MixtureDistribution:

    def __init__(self, dist_type='normal', mix_type='uniform', n_mixtures=1, dim=1):

        self.dist_type = dist_type
        self.mix_type = mix_type
        self.dists = [Distribution(self.dist_type, dim) for _ in range(n_mixtures)]
        self.sampling_prob = []

        if mix_type == 'uniform':

            self.sampling_prob = np.squeeze(np.ones((n_mixtures, 1))/n_mixtures)

        else:
            # Normal
            t = np.random.rand(n_mixtures, 1)
            self.sampling_prob = np.squeeze(t/np.linalg.norm(t, ord=1))

    def generate_samples(self, n_samples):

        n_mixtures = len(self.dists)
        counts = np.random.multinomial(n_samples, self.sampling_prob, 1)[0]

        samples = self.dists[0].generate_samples(counts[0])

        for i in range(1, n_mixtures):
            c_id = counts[i]

            this_samples = self.dists[i].generate_samples(c_id)

            samples = np.concatenate([samples, this_samples])

        return samples

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
            exp_prob = 0.0
            for k, mx in enumerate(self.sampling_prob):
                exp_prob += np.exp(self.dists[k].get_log_likelihood([sample]))*mx
            log_likelihood += np.log(exp_prob)

        return log_likelihood


class CirclesDatasetGenerator:

    def __init__(self, out_file, size=28, n_circles=1, modes=1, random_colors=False, random_sizes=False):

        self.size = size
        self.circles = n_circles
        self.has_random_colors = random_colors
        self.has_random_sizes = random_sizes
        self.modes = modes

        n_vars = n_circles * 2
        if random_colors:
            n_vars += n_circles * 3

        if random_sizes:
            n_vars += n_circles

        self.generator = MixtureDistribution('normal', 'uniform', modes, n_vars)

        self.random_color = [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) for i in
                        range(n_circles)]

        self.random_size = [np.random.randint(0, round(size * 0.25)) for i in range(n_circles)]
        self.save_generator(out_file)

    def generate_samples(self, n_samples):

        images = []

        samples = self.generator.generate_samples(n_samples)
        samples = (samples - np.min(samples, axis=0)) / (np.max(samples, axis=0) - np.min(samples, axis=0))

        for i in range(n_samples):

            this_image = np.ones((self.size, self.size, 3)) * 255
            for k in range(self.circles):
                sample = samples[i]

                # TODO: ~Abhi~

                if self.has_random_colors and self.has_random_sizes:
                    color = (
                    int(self.size * sample[6 * k + 3]), int(self.size * sample[6 * k + 4]), int(self.size * sample[6 * k + 5]))
                    circle_size = int(self.size * 0.25 * sample[6 * k + 2])
                    location = (int(self.size * sample[6 * k]), int(self.size * sample[6 * k + 1])
                    )

                elif self.has_random_sizes:
                    color = self.random_color[k]
                    circle_size = int(self.size * 0.25 * sample[3 * k + 2])
                    location = (int(self.size * sample[3 * k]), int(self.size * sample[3 * k + 1]))

                elif self.has_random_colors:
                    color = (
                    int(self.size * sample[5 * k + 2]), int(self.size * sample[5 * k + 3]), int(self.size * sample[5 * k + 4]))
                    circle_size = self.random_size[k]
                    location = (int(self.size * sample[5 * k]), int(self.size * sample[5 * k + 1])
                    )

                else:
                    color = self.random_color[k]
                    circle_size = self.random_size[k]
                    location = (int(self.size * sample[2 * k]), int(self.size * sample[2 * k + 1]))

                cv.circle(this_image, location, circle_size, color, -1)
            images.append(this_image)

        return np.array(images)/255.

    def generate_samples_to_directory(self, n_samples, output_directory):

        if output_directory:
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)

        samples = self.generator.generate_samples(n_samples)
        samples = (samples - np.min(samples, axis=0)) / (np.max(samples, axis=0) - np.min(samples, axis=0))

        for i in range(n_samples):

            this_image = np.ones((self.size, self.size, 3)) * 255
            for k in range(self.circles):
                sample = samples[i]

                if self.has_random_colors and self.has_random_sizes:
                    color = (
                        int(self.size * sample[6 * k + 3]), int(self.size * sample[6 * k + 4]),
                        int(self.size * sample[6 * k + 5]))
                    circle_size = int(self.size * 0.25 * sample[6 * k + 2])
                    location = (int(self.size * sample[6 * k]), int(self.size * sample[6 * k + 1]))

                elif self.has_random_sizes:
                    color = self.random_color[k]
                    circle_size = int(self.size * 0.25 * sample[3 * k + 2])
                    location = (int(self.size * sample[3 * k]), int(self.size * sample[3 * k + 1]))

                elif self.has_random_colors:
                    color = (
                        int(self.size * sample[5 * k + 2]), int(self.size * sample[5 * k + 3]),
                        int(self.size * sample[5 * k + 4]))
                    circle_size = self.random_size[k]
                    location = (int(self.size * sample[5 * k]), int(self.size * sample[5 * k + 1]))

                else:
                    color = self.random_color[k]
                    circle_size = self.random_size[k]
                    location = (int(self.size * sample[2 * k]), int(self.size * sample[2 * k + 1]))

                cv.circle(this_image, location, circle_size, color, -1)

                cv.imwrite(os.path.join(output_directory, '%d.jpg' % (i)), this_image)

    def save_generator(self, out_file):

        with open(out_file, 'wb') as of:
            pickle.dump(self, of, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":

    generator = CirclesDatasetGenerator(size=256, n_circles=4, random_colors=True, random_sizes=True, modes=20)
    x = generator.generate_samples(100)
    print(x)
    generator.generate_samples_to_directory(50, './dataset_1')
    generator.save_generator('./generator1.pickle')
