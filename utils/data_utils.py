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
        self.no_of_pixels = 28 * 28
        self.images = np.reshape(images, (self.num_datapts, self.no_of_pixels)) / 256.  # Flatten the image and normalize
        self.labels = labels

        self.minibatch_size = minibatch_size
        assert self.minibatch_size <= self.images.shape[0], "Minibatch must be less than the number of data points"
        self.epochs_completed = 0
        self.current_dataset_index = 0

    def shuffle_dataset(self):
        images_and_labels = np.hstack((self.images, np.expand_dims(self.labels, 1)))
        np.random.shuffle(images_and_labels)
        self.images = images_and_labels[:, :self.no_of_pixels]
        self.labels = images_and_labels[:, self.no_of_pixels]

    def next_batch(self, shuffle_every_epoch=True, images_only=True, minibatch_size=None):
        """
        Returns the next minibatch
        :return: np.ndarray, shape (batch_size, data_shape)
        """
        if minibatch_size is None: minibatch_size = self.minibatch_size

        current_index = self.current_dataset_index
        next_index = self.current_dataset_index + minibatch_size
        if next_index < self.num_datapts:  # next_index still points within the range of data points
            self.current_dataset_index = next_index
            if images_only:
                return self.images[current_index: next_index]
            else:
                return (self.images[current_index: next_index], self.labels[current_index: next_index])
        else:
            self.current_dataset_index = next_index % self.num_datapts
            self.epochs_completed += 1
            print("Completed {} epochs".format(self.epochs_completed))
            first_sub_batch_images = self.images[current_index:]  # The remainder of the current set of data points
            first_sub_batch_labels = self.labels[current_index:]
            if shuffle_every_epoch:
                self.shuffle_dataset()

            if images_only:
                return np.vstack((first_sub_batch_images, self.images[:self.current_dataset_index]))
            else:
                return (np.vstack((first_sub_batch_images, self.images[:self.current_dataset_index])),
                        np.vstack((first_sub_batch_labels, self.labels[:self.current_dataset_index])))