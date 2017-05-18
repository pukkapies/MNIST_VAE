import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib import gridspec
import numpy as np
import os
import itertools
import matplotlib.animation as animation


def save_plot(outdir, title):
    if not os.path.exists(outdir): os.makedirs(outdir)
    plt.savefig(os.path.join(outdir, title), bbox_inches="tight")


def plot_dataset_examples(x_in, labels=None, n=20, save=True, outdir='.'):
    """
    Plot some dataset image examples
    :param x_in: (batch_size, 28, 28) dataset images
    :param labels: Optional labels
    :param n: Number of plots. Make it divisible by 4
    :param save: Option to save
    :param outdir: Directory to save it to
    :return:
    """
    assert n % 4 == 0

    plt.figure()
    for i in range(1, n+1):
        fig = plt.subplot(4, n//4, i)
        plt.imshow(x_in[i, :, :], cmap="Greys")
        fig.get_xaxis().set_visible(False)
        fig.get_yaxis().set_visible(False)
        if labels is not None:
            fig.set_title("Label = {}".format(int(labels[i])))
    if save:
        save_plot(outdir, "Dataset examples.png")
    plt.close()

def plot_reconstruction(x_in, x_reconstructed, n=10, save=True, outdir="."):
    """
    Plot VAE reconstructions of given inputs
    :param model: Trained VAE object
    :param x_in: (batch_size, 28, 28) ground truth images
    :param x_reconstructed: (batch_size, 28, 28) VAE reconstructions
    :param n: Number of plots to make (constrained by the batch_size)
    :param save: Bool. Option to save
    :param outdir: Directory to save to
    :return:
    """
    n = min(n, x_in.shape[0])

    dim = 28  # Each image dimension (square image)

    def drawSubplot(x_, ax_):
        plt.imshow(x_, cmap="Greys")
        ax_.get_xaxis().set_visible(False)
        ax_.get_yaxis().set_visible(False)

    for i, x in enumerate(x_in[:n], 1):
        # display original
        ax = plt.subplot(n, 2, 2*i - 1)
        drawSubplot(x, ax)

    for i, x in enumerate(x_reconstructed[:n], 1):
        # display reconstruction
        ax = plt.subplot(n, 2, 2*i)
        drawSubplot(x, ax)

    # plt.show()
    if save:
        save_plot(outdir, "Model reconstructions.png")

    plt.close()


def plot_in_latent_space(model, x_in, labels, save=True, outdir="."):
    zTs = model.encode(x_in)
    ys, xs = zTs.T

    plt.figure()
    plt.title("Latent encodings of VAE")

    colours = ['b', 'c', 'y', 'm', 'r', 'k', 'g', (0.2, 0.4, 0.6), (0.8, 0.3, 0.5), (0.1, 0.1, 0.5)]

    scatter_pts = []  # To store a list of np.arrays for each digit
    for i in range(10):
        idx = (labels == i)
        scatter_pts.append(plt.scatter(xs[idx], ys[idx], color=colours[i], alpha=0.8))

    plt.legend(tuple([scatter_object for scatter_object in scatter_pts]), (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), loc='upper right')

    # plt.show()
    if save:
        save_plot(outdir, "Latent encodings of VAE.png")
    plt.close()


def plot_generation(model, z_in, x_in, labels, save=True, outdir='.', filename="Example generations.png"):
    """
    Horribly non-modular
    :param model:
    :param z_in:
    :param latent_fig:
    :param save:
    :param outdir:
    :return:
    """
    assert z_in.shape[0] == 1

    generation = model.decode(z_in)
    generation = generation.reshape(1, 28, 28)

    zTs = model.encode(x_in)
    ys, xs = zTs.T

    plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1])
    subplt = plt.subplot(gs[0])
    subplt.set_title('Latent space')

    classes = set(labels)
    colours = ['b', 'c', 'y', 'm', 'r', 'k', 'g', (0.2, 0.4, 0.6), (0.8, 0.3, 0.5), (0.1, 0.1, 0.5)]

    scatter_pts = []  # To store a list of np.arrays for each digit
    for i, class_ in enumerate(classes):
        idx = (labels == i)
        scatter_pts.append(plt.scatter(xs[idx], ys[idx], color=colours[i], alpha=0.2))

    plt.legend(tuple([scatter_object for scatter_object in scatter_pts]), (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
               loc='upper right')

    plt.plot(z_in[0, 0], z_in[0, 1], 'r^')  # 2D latent space only!

    subplt2 = plt.subplot(gs[1])
    subplt2.set_title("VAE generation")

    plt.imshow(generation[0, :, :], cmap="Greys")

    if save:
        save_plot(outdir, filename)
    plt.close()

def plot_generation_movie(model, z_in, save=True, outdir='.', filename="Generation movie.mp4"):
    assert z_in.shape[1] == 2

    num_frames = z_in.shape[0]

    mov = plt.figure(figsize=(12, 6))

    frames = []
    for frame in range(num_frames):

        z = z_in[frame, :]
        z = np.expand_dims(z, 0)
        generation = model.decode(z)
        generation = generation.reshape(1, 28, 28)
        generation = generation[0, :, :]
        f = plt.imshow(generation, cmap="Greys")
        plt.grid('off')
        plt.title('VAE generations')
        frames.append((f, ))

    anim = animation.ArtistAnimation(mov, frames, interval=100)
    if save:
        if not os.path.exists(outdir): os.makedirs(outdir)
        anim.save(outdir + filename)

    plt.close()





def plot_generation_movie_subplots(model, z_in, x_in, labels, save=True, outdir='.', filename="Generation movie.mp4"):
    #TODO: Make this work!
    assert z_in.shape[1] == 2

    num_frames = z_in.shape[0]

    mus, _ = model.encode(x_in)
    ys, xs = mus.T

    classes = set(labels)
    colours = ['b', 'c', 'y', 'm', 'r', 'k', 'g', (0.2, 0.4, 0.6), (0.8, 0.3, 0.5), (0.1, 0.1, 0.5)]

    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)

    frames = []
    for frame in range(num_frames):
        for i, class_ in enumerate(classes):
            idx = (labels == i)
            ax1.scatter(xs[idx], ys[idx], color=colours[i], alpha=0.2)
        im1 = ax1.plot(z_in[frame, 0], z_in[frame, 1], 'r^')  # 2D latent space only!

        z = z_in[frame, :]
        z = np.expand_dims(z, 0)
        generation = model.decode(z)
        generation = generation.reshape(1, 28, 28)
        generation = generation[0, :, :]
        im2 = ax2.imshow(generation, cmap="Greys")

        frames.append([im1, im2])






    # mov = plt.figure(figsize=(12, 6))
    # gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1])

    anim = animation.ArtistAnimation(fig, frames, interval=100)
    plt.show()
    if save:
        if not os.path.exists(outdir): os.makedirs(outdir)
        anim.save(outdir + filename)

    plt.close()











