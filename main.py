from __future__ import absolute_import, print_function, division
import os
import numpy as np
import tensorflow as tf
from get_mnist import save_npy_datasets
from layers import FeedForward, Dense
from vae import VAE
from data_utils import DatasetFeed
from tensorflow.python.ops.init_ops import variance_scaling_initializer

# MODEL_TO_RESTORE = 'training/saved_models/170324_1035model-100'
MODEL_TO_RESTORE = None


IMAGE_SIZE = 28*28
LATENT_DIM = 2

ENCODER_ARCH = [600, 400]  # Takes IMG_SIZE units as input
DECODER_ARCH = [400, 600, IMAGE_SIZE]  # Takes LATENT_DIM units as input

MINIBATCH_SIZE = 128

MAX_ITER = 100

curr_path = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':

    # Get and load data
    if not os.path.exists(curr_path + '/MNIST/'):
        save_npy_datasets()
    training_images = np.load(curr_path + "/MNIST/train_images.npy")
    training_labels = np.load(curr_path + "/MNIST/train_labels.npy")
    validation_images = np.load(curr_path + "/MNIST/validation_images.npy")
    validation_labels = np.load(curr_path + "/MNIST/validation_labels.npy")
    test_images = np.load(curr_path + "/MNIST/test_images.npy")
    test_labels = np.load(curr_path + "/MNIST/test_labels.npy")

    train_data = DatasetFeed(training_images, training_labels, MINIBATCH_SIZE)

    # Define encoder network
    encoder = FeedForward(scope="encoder", sizes=ENCODER_ARCH, nonlinearity=tf.nn.elu,
                          initializer=variance_scaling_initializer(scale=2.0, mode="fan_in", distribution="normal"))
    decoder = FeedForward(scope="decoder", sizes=DECODER_ARCH,
                          nonlinearity=[tf.nn.elu]*(len(DECODER_ARCH)-1) + [tf.nn.sigmoid],
                          initializer=variance_scaling_initializer(scale=2.0, mode="fan_in", distribution="normal"))


    vae = VAE(encoder, decoder, LATENT_DIM, d_hyperparams={}, model_to_restore=None)
    vae.train(train_data, max_iter=MAX_ITER, max_epochs=np.inf, verbose=True, save=True)

    # vae = VAE(encoder, decoder, LATENT_DIM, model_to_restore=MODEL_TO_RESTORE)
    # vae.test(train_data, 10)  # Test the model on 10 minibatches