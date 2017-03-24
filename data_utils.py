from __future__ import division
import numpy as np


class DatasetFeed(object):

    def __init__(self, images, labels, minibatch_size):
        """
        Initializes the object for feeding the MNIST data
        :param images: np.array (no. of datapts, 28, 28)
        :param labels: np.array (no. of datapts, )
        :param minibatch_size: Int
        """
        self.num_datapts = images.shape[0]
        self.images = np.reshape(images, (self.num_datapts, 28*28)) / 256.  # Flatten the image and normalize
        self.labels = labels

        self.minibatch_size = minibatch_size
        assert self.minibatch_size <= self.images.shape[0], "Minibatch must be less than the number of data points"
        self.epochs_completed = 0
        self.current_dataset_index = 0

    def next_batch(self, shuffle_every_epoch=True):
        """
        Returns the next minibatch
        :return: np.ndarray, shape (batch_size, data_shape)
        """
        current_index = self.current_dataset_index
        next_index = self.current_dataset_index + self.minibatch_size
        if next_index < self.num_datapts:  # next_index still points within the range of data points
            self.current_dataset_index = next_index
            return np.asarray(self.images[current_index: next_index])
        else:
            self.current_dataset_index = next_index % self.num_datapts
            self.epochs_completed += 1
            print("Completed {} epochs".format(self.epochs_completed))
            first_sub_batch = self.images[current_index:]  # The remainder of the current set of data points
            if shuffle_every_epoch:
                np.random.shuffle(self.images)

            return np.vstack((first_sub_batch, self.images[:self.current_dataset_index]))