import tensorflow as tf
from functools import partial, reduce
from tensorflow.python.ops.init_ops import Zeros


def compose2(f, g):
    return lambda x: f(g(x))


def compose(*args):
    return reduce(compose2, args)


def identity(x):
    return x


def compose_all(*args):
    """Util for multiple function composition

    i.e. composed = composeAll([f, g, h])
         composed(x) # == f(g(h(x)))
    """
    if len(*args) == 0:
        return identity
    return partial(reduce, compose)(*args)


class Dense(object):
    """Fully-connected layer. Can be applied to Tensors of shape (batch_size, n_inputs)"""
    def __init__(self, scope="dense_layer", size=None, nonlinearity=tf.identity, initializer=None):
        assert size, "Must specify layer size (num nodes)"
        assert initializer, "Must specify an initialiser for Dense layer"
        self.scope = scope
        self.size = size
        self.nonlinearity = nonlinearity
        self.initializer = initializer
        self.settings = {'type': 'Dense', 'layer_size': size, 'nonlinearity': nonlinearity.__name__,
                         'scope': scope}

    def __call__(self, x):
        with tf.variable_scope(self.scope):
            self.w = tf.get_variable(name='W', shape=(x.get_shape()[-1].value, self.size), dtype=tf.float32,
                                     initializer=self.initializer)
            self.b = tf.get_variable(name='b', shape=[self.size], dtype=tf.float32, initializer=Zeros())
            return self.nonlinearity(tf.matmul(x, self.w) + self.b)


class FeedForward(object):
    """Feedforward network/MLP"""
    def __init__(self, scope="dense_layer", sizes=None, initializer=None, nonlinearity=None):
        """
        Initializer
        :param scope: string, or list of strings. If a list, then must be the same length as sizes, and each
                        string entry will be used as the scope for each hidden layer
        :param sizes: list of layer sizes. First entry is first hidden layer size, last layer is output.
                        The input size is obtained when the class is called, so is not needed in the initialisation.
        :param nonlinearity: Nonlinear activation function to be used in the MLP, or a list of activation functions
                        of length equal to sizes
        """
        assert sizes, "Need to specify layer sizes for Feedforward architecture"
        assert nonlinearity, "Need to specify nonlinearity for Feedforward architecture"
        self.settings = {'type': 'FeedForward', 'layer_sizes': sizes,
                         'nonlinearity': [nl.__name__ for nl in nonlinearity], 'scope': scope}
        self.sizes = sizes[::-1]  # !! Reverse the order for function composition
        assert type(nonlinearity) == list and len(nonlinearity) == len(sizes)
        self.nonlinearity = nonlinearity[::-1]  # Reverse order to match the layers
        self.initializer = initializer
        assert (type(scope) == str) or (type(scope) == list)
        if type(scope) == list:
            assert len(scope) == len(sizes)
            self.scope = scope[::-1]  # Reverse order to match the layers
        else:
            self.scope = [scope_name + '_layer' + str(i) for i, scope_name in enumerate([scope] * len(sizes))][::-1]

    def __call__(self, x):
        layer_fns = [Dense(self.scope[i], hidden_size, nonlinearity=self.nonlinearity[i], initializer=self.initializer)
                     for i, hidden_size in enumerate(self.sizes)]
        output = compose_all(layer_fns)(x)
        return output
