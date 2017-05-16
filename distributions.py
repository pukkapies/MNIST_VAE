import tensorflow as tf
import numpy as np


def gaussian_diag_logps(mean, logvar, sample):
    return -0.5 * (np.log(2 * np.pi) + logvar + tf.square(sample - mean) / tf.exp(logvar))


class DiagonalGaussian(object):

    def __init__(self, mean, logvar):
        self.mean = mean
        self.logvar = logvar

    def sample(self):
        noise = tf.random_normal(tf.shape(self.mean))
        sample = self.mean + tf.exp(0.5 * self.logvar) * noise
        return sample

    def logprob(self, sample):
        return gaussian_diag_logps(self.mean, self.logvar, sample)
