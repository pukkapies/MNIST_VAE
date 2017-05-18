from __future__ import absolute_import, print_function, division

import os

import numpy as np
import tensorflow as tf
from tensorflow.python.ops.init_ops import variance_scaling_initializer

from get_mnist import save_npy_datasets
from layers import FeedForward
from make_plots import plot_reconstruction, plot_in_latent_space, plot_dataset_examples, plot_generation, plot_generation_movie
from utils.data_utils import DatasetFeed
from vae import VAE

# MODEL_TO_RESTORE = 'training/saved_models/170518_1455/model-100000'
MODEL_TO_RESTORE = None


IMAGE_SIZE = 28*28
LATENT_DIM = 2

ENCODER_ARCH = [600, 400]  # Takes IMG_SIZE units as input
DECODER_ARCH = [400, 600, IMAGE_SIZE]  # Takes LATENT_DIM units as input

MINIBATCH_SIZE = 128

MAX_ITER = 500000
HYPERPARAMS = {'learning_rate': 5E-4}

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
    encoder = FeedForward(scope="encoder", sizes=ENCODER_ARCH, nonlinearity=[tf.nn.elu] * len(ENCODER_ARCH),
                          initializer=variance_scaling_initializer(scale=2.0, mode="fan_in", distribution="normal"))
    decoder = FeedForward(scope="decoder", sizes=DECODER_ARCH,
                          nonlinearity=[tf.nn.elu]*(len(DECODER_ARCH)-1) + [tf.nn.sigmoid],
                          initializer=variance_scaling_initializer(scale=2.0, mode="fan_in", distribution="normal"))


    if MODEL_TO_RESTORE is None:
        vae = VAE(encoder, decoder, LATENT_DIM, d_hyperparams=HYPERPARAMS, model_to_restore=None)
        vae.train(train_data, max_iter=MAX_ITER, max_epochs=np.inf, verbose=True, save=True)
    else:
        vae = VAE(encoder, decoder, LATENT_DIM, model_to_restore=MODEL_TO_RESTORE)

        train_data.shuffle_dataset()  # Just so plots are different every time
        vae.test(train_data, 10)  # Test the model on 10 minibatches

        ###################### MAKE PLOTS ##########################
        PLOT_DIR = vae.model_folder + 'plots/'

        train_data.shuffle_dataset()  # Just so plots are different every time

        # Plot some dataset examples
        data_minibatch_images, data_minibatch_labels = train_data.next_batch(images_only=False)
        print("Plotting some dataset examples...", end="")
        data_minibatch_images = np.reshape(data_minibatch_images, (MINIBATCH_SIZE, 28, 28))  # Reshape inputs and outputs for plotting
        plot_dataset_examples(data_minibatch_images, labels=None, outdir=PLOT_DIR)
        print('done.')

        # Reconstruction plot
        data_minibatch = train_data.next_batch()
        print("Generating reconstruction plots from minibatch {}...".format(data_minibatch.shape), end="")
        vae_reconstruction = vae.sess.run(vae.vae_output, feed_dict={vae.input_ph: data_minibatch})

        data_minibatch = np.reshape(data_minibatch, (MINIBATCH_SIZE, 28, 28)) # Reshape inputs and outputs for plotting
        vae_reconstruction = np.reshape(vae_reconstruction, (MINIBATCH_SIZE, 28, 28))
        plot_reconstruction(data_minibatch, vae_reconstruction, n=8, outdir=PLOT_DIR)
        print('done.')

        if vae.settings['latent_dim'] == 2:
            # Plots of encodings in latent space
            data_minibatch_images, data_minibatch_labels = train_data.next_batch(images_only=False, minibatch_size=1000)
            print("Generating plots for latent space encodings...", end="")
            plot_in_latent_space(vae, data_minibatch_images, labels=data_minibatch_labels, outdir=PLOT_DIR)
            print('done.')

            # plt.figure()
            # print(data_minibatch_images.shape)
            # data_reshaped = data_minibatch_images.reshape(1000, 28, 28)
            # for j in range(4):
            #     plt.subplot(2, 2, j+1)
            #     plt.imshow(data_reshaped[j,:, :])
            #     print('Label: ', data_minibatch_labels[j])
            # plt.show()


            print("Making generated figures for given latent variable...", end="")
            # plot_generation(vae, np.array([[-0.3, -0.3]]), data_minibatch_images, data_minibatch_labels,
            #                 outdir=PLOT_DIR, filename="Example generations 1.png")
            # plot_generation(vae, np.array([[-2, -3]]), data_minibatch_images, data_minibatch_labels,
            #                 outdir=PLOT_DIR, filename="Example generations 2.png")
            # plot_generation(vae, np.array([[1.4, 2.5]]), data_minibatch_images, data_minibatch_labels,
            #                 outdir=PLOT_DIR, filename="Example generations 3.png")
            # plot_generation(vae, np.array([[1.7, -0.1]]), data_minibatch_images, data_minibatch_labels,
            #                 outdir=PLOT_DIR, filename="Example generations 4.png")
            # plot_generation(vae, np.array([[2.5, 0]]), data_minibatch_images, data_minibatch_labels,
            #                 outdir=PLOT_DIR, filename="Example generations 5.png")

            plot_generation(vae, np.array([[-0.3, -0.3]]), data_minibatch_images, data_minibatch_labels,
                            outdir=PLOT_DIR, filename="Example generations 1.png")
            plot_generation(vae, np.array([[-2, -3]]), data_minibatch_images, data_minibatch_labels,
                            outdir=PLOT_DIR, filename="Example generations 2.png")
            plot_generation(vae, np.array([[1.4, 2.5]]), data_minibatch_images, data_minibatch_labels,
                            outdir=PLOT_DIR, filename="Example generations 3.png")
            plot_generation(vae, np.array([[1.7, -0.1]]), data_minibatch_images, data_minibatch_labels,
                            outdir=PLOT_DIR, filename="Example generations 4.png")
            plot_generation(vae, np.array([[2.5, 0]]), data_minibatch_images, data_minibatch_labels,
                            outdir=PLOT_DIR, filename="Example generations 5.png")

            print('done.')

        print("Making movie...", end="")
        number_of_cycles = 10
        circle_param = np.linspace(0, number_of_cycles*2*np.pi, num=number_of_cycles * 120)
        zs_list = [np.array([3 * np.sin(0.3 * circle_param[i]) * np.cos(circle_param[i]),
                             3 * np.sin(0.3 * circle_param[i]) * np.sin(circle_param[i])]) for i in range(circle_param.shape[0])]
        zs = np.asarray(zs_list)
        plot_generation_movie(vae, zs, outdir=PLOT_DIR)
        # plot_generation_movie_subplots(vae, zs, data_minibatch_images, data_minibatch_labels, save=False,
        #                                outdir=PLOT_DIR, filename="Generation movie subplots")
        print('done.')
