import numpy as np
import tensorflow as tf
from tensorflow.python.ops.init_ops import Zeros


def get_linear_ar_mask(n_in, n_out, zerodiagonal=False):
    assert n_in % n_out == 0 or n_out % n_in == 0, "%d - %d" % (n_in, n_out)

    mask = np.ones([n_in, n_out], dtype=np.float32)
    if n_out >= n_in:
        k = n_out / n_in
        for i in range(n_in):
            mask[i + 1:, i * k:(i + 1) * k] = 0
            if zerodiagonal:
                mask[i:i + 1, i * k:(i + 1) * k] = 0
    else:
        k = n_in / n_out
        for i in range(n_out):
            mask[(i + 1) * k:, i:i + 1] = 0
            if zerodiagonal:
                mask[i * k:(i + 1) * k:, i:i + 1] = 0
    return mask


class AR_Dense(object):
    """Autoregressive linear layer. Can be applied to Tensors of shape (batch_size, n_inputs)"""
    def __init__(self, size, initializer, zerodiagonal, scope="dense_layer", nonlinearity=tf.identity):
        self.scope = scope
        self.size = size
        self.nonlinearity = nonlinearity
        self.initializer = initializer
        self.zerodiagonal = zerodiagonal

    def __call__(self, x):
        with tf.variable_scope(self.scope):
            mask = get_linear_ar_mask(x.get_shape()[-1].value, self.size, zerodiagonal=self.zerodiagonal)
            w = tf.get_variable(name='W', shape=(x.get_shape()[-1].value, self.size), dtype=tf.float32,
                                     initializer=self.initializer)
            self.w = w * mask
            self.b = tf.get_variable(name='b', shape=[self.size], dtype=tf.float32, initializer=Zeros())
            return self.nonlinearity(tf.matmul(x, self.w) + self.b)
