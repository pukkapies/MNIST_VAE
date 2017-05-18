from __future__ import print_function, division
import os
import sys
from datetime import datetime
import numpy as np
import tensorflow as tf
from layers import Dense
from tensorflow.python.ops.init_ops import variance_scaling_initializer
from distributions import DiagonalGaussian
from ar_layers import AR_Dense

#TODO: Pass this in in the initialization somehow
IMAGE_SIZE = 28*28  # For MNIST


class VAE(object):
    """The Variational Autoencoder network. This class is intended to be a general implementation, where the
     encoder and decoder architectures are defined separately and passed in at initialisation.

    References
    ----------

    [1] Kingma & Welling (2013), Auto-Encoding Variational Bayes, http://arxiv.org/abs/1312.6114.

    [2] Rezende, Mohamed and Wierstra (2014), Stochastic backpropagation and approximate inference
    in deep generative models, https://arxiv.org/abs/1401.4082.
    """
    DEFAULTS = {
        "learning_rate": 1E-3
    }
    RESTORE_KEY = "to_restore"
    DEBUG_KEY = 'debug_'

    def __init__(self, encoder, decoder, latent_dim, d_hyperparams={}, scope='VAE', save_graph_def=True,
                 log_dir="./log/", analysis_dir=None, model_to_restore=None):
        """
        Initialiser
        :param encoder: Encoder architecture. Should be a callable object to apply to inputs
        :param decoder: Dencoder architecture. Should be a callable object to apply to latent variable
        :param latent_dim: Number of units for the latent space
        :param d_hyperparams: Optional hyperparameters to update VAE defaults
        :param scope: Scope for VAE
        :param save_graph_def: Bool. Option to write summaries for tensorboard
        :param log_dir: Logging directory for tensorboard
        :param analysis_dir: Folder to save various things for training analysis
        :param model_to_restore: Either None, or path to metagraph for restoring. If the model is to be restored,
               many of the former build options are ignored
        """
        self.sess = tf.Session()
        self.settings = VAE.DEFAULTS
        self.settings.update(**d_hyperparams)

        if model_to_restore is None:
            print("Building new VAE model")
            self.encoder = encoder
            self.decoder = decoder
            self.latent_dim = latent_dim
            self.scope = scope

            model_datetime = datetime.now().strftime(r"%y%m%d_%H%M")
            self.model_folder = './training/saved_models/' + model_datetime + '/'
            if not os.path.exists(self.model_folder): os.makedirs(self.model_folder)

            if analysis_dir is None:
                self.analysis_folder = self.model_folder + 'analysis/'
            else:
                self.analysis_folder = analysis_dir
            if not os.path.exists(self.analysis_folder): os.makedirs(self.analysis_folder)

            # Build the graph
            handles = self._build_graph()
            for handle in handles:
                tf.add_to_collection(VAE.RESTORE_KEY, handle)
            self.sess.run(tf.global_variables_initializer())

            # unpack handles for tensor ops to feed or fetch
            self.unpack_handles(handles)
        else:
            print('Restoring model: ', model_to_restore)
            self.model_folder = '/'.join((model_to_restore.split('/')[:-1])) + '/'

            # Rebuild the graph
            meta_graph = os.path.abspath(model_to_restore)
            tf.train.import_meta_graph(meta_graph + ".meta").restore(self.sess, meta_graph)

            handles = self.sess.graph.get_collection(VAE.RESTORE_KEY)

            print("Restored handles: ", handles)
            self.unpack_handles(handles)

        if save_graph_def:  # tensorboard
            self.logger = tf.summary.FileWriter(log_dir, self.sess.graph)

    def unpack_handles(self, handles):
        (self.input_ph, self.ar_mean, self.ar_logsigma, self.vae_output, self.z_,
         self.x_reconstructed_, self.cost, self.train_op, self.cost_no_KL, self.kl_loss, self.global_step) = handles

    @property
    def step(self):
        """Train step"""
        return self.global_step.eval(session=self.sess)

    def save_model(self, outdir):
        """Saves the model if a self.saver object exists"""
        try:
            outfile = outdir + 'model'
            self.saver.save(self.sess, outfile, global_step=self.step)
        except AttributeError:
            print("Failed to save model at step {}".format(self.step))
            return

    def _build_graph(self):
        with tf.variable_scope(self.scope) as graph_scope:
            input_ph = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE])  # (batch_size, n_inputs)
            print("input shape: ", input_ph.get_shape())

            global_step = tf.Variable(0, trainable=False, name="global_step")

            prelatent_layer = self.encoder(input_ph)

            z_mean = Dense(scope="z_mean", size=self.latent_dim,
                           initializer=variance_scaling_initializer(scale=2.0, mode="fan_in", distribution="normal"))(prelatent_layer)
            z_log_sigma = Dense(scope="z_log_sigma", size=self.latent_dim,
                                initializer=variance_scaling_initializer(scale=2.0, mode="fan_in", distribution="normal"))(prelatent_layer)

            prior = DiagonalGaussian(tf.zeros(self.latent_dim), tf.ones(self.latent_dim))  # N(0, 1)
            posterior = DiagonalGaussian(z_mean, z_log_sigma)

            tf.add_to_collection(VAE.DEBUG_KEY + 'z_mean', z_mean)
            tf.add_to_collection(VAE.DEBUG_KEY + 'z_log_sigma', z_log_sigma)

            print("Finished setting up encoder")
            print([var._variable for var in tf.global_variables()])

            print('z_mean shape: ', z_mean.get_shape())
            print('z_log_sigma shape: ', z_log_sigma.get_shape())

            z = posterior.sample()
            print("Posterior sample shape: ", z.get_shape())

            # z = self.sampleGaussian(z_mean, z_log_sigma)  # (batch_size, latent_dim)
            self.z = z ## TO REMOVE - DEBUGGING ONLY

            tf.add_to_collection(VAE.DEBUG_KEY + 'z', z)

            logqs = posterior.logprob(z)

            # IAF Posterior
            # Create two AR layers
            first_AR_Dense = AR_Dense(self.latent_dim,
                                      variance_scaling_initializer(scale=2.0, mode="fan_avg", distribution="normal"),
                                      zerodiagonal=False, scope='ar_layer1', nonlinearity=tf.nn.elu)
            second_AR_Dense_to_mean = AR_Dense(self.latent_dim,
                                               variance_scaling_initializer(scale=2.0, mode="fan_avg", distribution="normal"),
                                               zerodiagonal=True, scope='ar_layer2_mean')
            second_AR_Dense_to_logsigma = AR_Dense(self.latent_dim,
                                                   variance_scaling_initializer(scale=2.0, mode="fan_avg", distribution="normal"),
                                                   zerodiagonal=True, scope='ar_layer2_logsd')

            hidden_AR_layer = first_AR_Dense(z)
            ar_mean = second_AR_Dense_to_mean(hidden_AR_layer)
            ar_logsigma = second_AR_Dense_to_logsigma(hidden_AR_layer)

            tf.add_to_collection(VAE.DEBUG_KEY + 'ar_mean', ar_mean)
            tf.add_to_collection(VAE.DEBUG_KEY + 'ar_logsigma', ar_logsigma)

            # self.ar_mean = ar_mean  # for debugging
            # self.ar_logsigma = ar_logsigma  # for debugging

            z = (z - ar_mean) / tf.exp(ar_logsigma)
            print("Post AR z.shape", z.get_shape())

            tf.add_to_collection(VAE.DEBUG_KEY + 'z', z)

            self.first_ar_layer_weights = first_AR_Dense.w  # for debugging
            self.second_ar_layer_weights_mean = second_AR_Dense_to_mean.w  # for debugging
            self.second_ar_layer_weights_logsigma = second_AR_Dense_to_logsigma.w  # for debugging

            tf.add_to_collection(VAE.DEBUG_KEY + 'ar_layer1_w', first_AR_Dense.w)
            tf.add_to_collection(VAE.DEBUG_KEY + 'ar_layer1_b', first_AR_Dense.b)
            
            tf.add_to_collection(VAE.DEBUG_KEY + 'ar_layer2_mean_w', second_AR_Dense_to_mean.w)
            tf.add_to_collection(VAE.DEBUG_KEY + 'ar_layer2_logsd_w', second_AR_Dense_to_logsigma.w)
            tf.add_to_collection(VAE.DEBUG_KEY + 'ar_layer2_mean_b', second_AR_Dense_to_mean.b)
            tf.add_to_collection(VAE.DEBUG_KEY + 'ar_layer2_logsd_b', second_AR_Dense_to_logsigma.b)

            logqs += ar_logsigma
            logps = prior.logprob(z)
            kl_obj = logqs - logps  # (batch_size, latent_dim)

            kl_obj = tf.reduce_sum(kl_obj, [1])

            vae_output = self.decoder(z)  # (batch_size, n_outputs)
            print("vae output shape: ", vae_output.get_shape())

            print("Finished setting up decoder")
            print([var._variable for var in tf.global_variables()])

            ##########################################################################

            # Set up gradient calculation and optimizer
            # rec_loss = tf.losses.sigmoid_cross_entropy(input_ph, vae_output)
            rec_loss = self.crossEntropy(vae_output, input_ph)
            print('rec_loss shape:', rec_loss.get_shape())  # THINK I MIGHT NEED TO TAKE A MEAN HERE
            print('kl_obj shape:', kl_obj.get_shape())

            # # Kullback-Leibler divergence: mismatch b/w approximate vs. imposed/true posterior
            # kl_loss = self.kullback_leibler_diag_gaussian(z_mean, z_log_sigma)
            # print('kl_loss shape:', kl_loss.get_shape())

            with tf.name_scope("cost"):
                # average over minibatch
                cost = tf.reduce_mean(kl_obj + rec_loss, name="vae_cost")
                cost_no_KL = tf.reduce_mean(rec_loss, name='rec_cost')
                cost_KL = tf.reduce_mean(kl_obj, name="KL_cost")

            print("Defined loss functions")

            # optimization
            optimizer = tf.train.AdamOptimizer(learning_rate=self.settings['learning_rate'])
            tvars = tf.trainable_variables()
            grads_and_vars = optimizer.compute_gradients(cost, tvars)
            clipped = [(tf.clip_by_value(grad, -5, 5), tvar)  # gradient clipping
                       for grad, tvar in grads_and_vars]
            train_op = optimizer.apply_gradients(clipped, global_step=global_step, name="minimize_cost")
            print("Defined training ops")

            print([var._variable for var in tf.global_variables()])

            # ops to directly explore latent space
            # defaults to prior z ~ N(0, I)
            with tf.name_scope("latent_in"):
                z_ = tf.placeholder_with_default(tf.random_normal([1, self.latent_dim]), shape=[1, self.latent_dim],
                                                 name="latent_in")
            graph_scope.reuse_variables()  # No new variables should be created from this point on
            x_reconstructed_ = self.decoder(z_)

            return (input_ph, ar_mean, ar_logsigma, vae_output, z_, x_reconstructed_, cost,
                    train_op, cost_no_KL, cost_KL, global_step)

    def sampleGaussian(self, mu, log_sigma):
        """Draw sample from Gaussian with given shape, subject to random noise epsilon"""
        # TODO: Apply more draws as an option. Reference [1] states one is enough if minibatch size > 100
        with tf.name_scope("sample_gaussian"):
            # reparameterization trick
            epsilon = tf.random_normal(tf.shape(log_sigma), name="epsilon")
            self.epsilon = epsilon  ##TO REMOVE - DEBUGGING ONLY
            return mu + epsilon * tf.exp(log_sigma) # N(mu, I * sigma**2)

    def crossEntropy(self, obs, actual, offset=1e-7):
        """Binary cross-entropy, per training example"""
        # (tf.Tensor, tf.Tensor, float) -> tf.Tensor
        with tf.name_scope("cross_entropy"):
            # bound by clipping to avoid nan
            obs_ = tf.clip_by_value(obs, offset, 1 - offset)
            return -tf.reduce_sum(actual * tf.log(obs_) +
                                  (1 - actual) * tf.log(1 - obs_), 1)


    def kullback_leibler_diag_gaussian(self, mu, log_sigma):
        """
        Calculates the KL-divergence between the approximate posterior and prior.
        For a diagonal approximate posterior and standard Gaussian prior, this can be calculated exactly.
        :param mu: Mean of approximate posterior: shape (batch_size, latent_dim)
        :param log_sigma: Log-sd of approximate posterior: shape (batch_size, latent_dim)
        :return: KL-divergence between approximate posterior and prior
        """
        with tf.name_scope("KL_divergence"):
            # = -0.5 * (1 + log(sigma**2) - mu**2 - sigma**2)
            return -0.5 * tf.reduce_sum(1 + 2 * log_sigma - mu**2 - tf.exp(2 * log_sigma), 1)

    def encode(self, x):
        """Probabilistic encoder from inputs to latent distribution parameters;
        a.k.a. inference network q(z|x)
        """
        # np.array -> [float, float]
        feed_dict = {self.input_ph: x}
        return self.sess.run([self.ar_mean, self.ar_logsigma], feed_dict=feed_dict)

    def decode(self, zs=None):
        """Generative decoder from latent space to reconstructions of input space;
        a.k.a. generative network p(x|z)
        """
        # (np.array | tf.Variable) -> np.array
        feed_dict = dict()
        if zs is not None:
            is_tensor = lambda x: hasattr(x, "eval")
            zs = (self.sess.run(zs) if is_tensor(zs) else zs) # coerce to np.array
            feed_dict.update({self.z_: zs})
        # else, zs defaults to draw from conjugate prior z ~ N(0, I)
        return self.sess.run(self.x_reconstructed_, feed_dict=feed_dict)

    def vae(self, x):
        """End-to-end variational autoencoder"""
        return self.decode(self.sampleGaussian(*self.encode(x)))

    def train(self, dataset, max_iter=np.inf, max_epochs=np.inf, verbose=True, save=True):
        if save:
            self.saver = tf.train.Saver(tf.global_variables())

        total_cost_history = np.array([])
        KL_cost_history = np.array([])
        reconstruction_cost_history = np.array([])

        outdir = self.model_folder
        self.accumulated_cost = 0

        now = datetime.now().isoformat()[11:]
        print("------- Training begin: {} -------\n".format(now))

        while True:
            try:
                x = dataset.next_batch()  # (batch_size, n_inputs)

                feed_dict = {self.input_ph: x}
                
                fetches = [self.vae_output, self.cost, self.kl_loss, self.cost_no_KL, self.global_step, self.train_op]
                x_reconstructed, cost, kl_loss, rec_loss, i, _ = self.sess.run(fetches, feed_dict=feed_dict)

                # print("AR first layer weight matrix:")
                # print(self.sess.run([self.first_ar_layer_weights], feed_dict=feed_dict))
                # print("AR second layer weight matrix to mean:")
                # print(self.sess.run([self.second_ar_layer_weights_mean], feed_dict=feed_dict))
                # print("AR second layer weight matrix to logsigma:")
                # print(self.sess.run([self.second_ar_layer_weights_logsigma], feed_dict=feed_dict))
                #
                # print("AR mean:")
                # print(self.sess.run([self.ar_mean], feed_dict=feed_dict))
                # print("AR logsigma:")
                # print(self.sess.run([self.ar_logsigma], feed_dict=feed_dict))

                self.accumulated_cost += cost

                total_cost_history = np.hstack((total_cost_history, np.array([float(cost)])))
                KL_cost_history = np.hstack((KL_cost_history, np.array([float(kl_loss)])))
                reconstruction_cost_history = np.hstack((reconstruction_cost_history, np.array([float(rec_loss)])))

                if i % 5000 == 0:
                    # SAVE ALL MODEL PARAMETERS
                    self.save_model(outdir)

                if i % 500 == 0:
                    # SAVE COSTS FOR LEARNING CURVES
                    np.save(self.analysis_folder + 'total_cost.npy', total_cost_history)
                    np.save(self.analysis_folder + 'KL_cost.npy', KL_cost_history)
                    np.save(self.analysis_folder + 'reconstruction_cost.npy', reconstruction_cost_history)

                if i % 1000 == 0:
                    # PRINT PROGRESS
                    print("Step {}-> cost for this minibatch: {}".format(i, cost))
                    print("   minibatch KL_cost = {}, reconst = {}".format(np.mean(kl_loss), np.mean(rec_loss)))

                if i >= max_iter or dataset.epochs_completed >= max_epochs:
                    print("final avg cost (@ step {} = epoch {}): {}".format(
                        i, dataset.epochs_completed, self.accumulated_cost / i))
                    now = datetime.now().isoformat()[11:]
                    print("------- Training end: {} -------\n".format(now))

                    self.save_model(outdir)
                    try:
                        self.logger.flush()
                        self.logger.close()
                    except(AttributeError): # not logging
                        pass
                    return cost

            except(KeyboardInterrupt):
                print("final avg cost (@ step {} = epoch {}): {}".format(
                    i, dataset.epochs_completed, self.accumulated_cost / i))
                now = datetime.now().isoformat()[11:]
                print("------- Training end: {} -------\n".format(now))
                sys.exit(0)

    def test(self, dataset, iterations):
        print("Some example test minibatch costs: ")
        for _ in range(iterations):
            x = dataset.next_batch()  # (batch_size, n_inputs)
            feed_dict = {self.input_ph: x}

            fetches = [self.vae_output, self.cost, self.kl_loss, self.cost_no_KL, self.global_step]
            x_reconstructed, cost, kl_loss, rec_loss, i = self.sess.run(fetches, feed_dict=feed_dict)

            print(cost)

            z_mean = self.sess.graph.get_collection(VAE.DEBUG_KEY + 'z_mean')
            z_log_sigma = self.sess.graph.get_collection(VAE.DEBUG_KEY + 'z_log_sigma')
            z = self.sess.graph.get_collection(VAE.DEBUG_KEY + 'z')
            ar_mean = self.sess.graph.get_collection(VAE.DEBUG_KEY + 'ar_mean')
            ar_logsigma = self.sess.graph.get_collection(VAE.DEBUG_KEY + 'ar_logsigma')
            z = self.sess.graph.get_collection(VAE.DEBUG_KEY + 'z')
            ar_layer1_w = self.sess.graph.get_collection(VAE.DEBUG_KEY + 'ar_layer1_w')
            ar_layer2_mean_w = self.sess.graph.get_collection(VAE.DEBUG_KEY + 'ar_layer2_mean_w')
            ar_layer2_logsd_w = self.sess.graph.get_collection(VAE.DEBUG_KEY + 'ar_layer2_logsd_w')

            # print(self.sess.run(z_mean, feed_dict=feed_dict))
            print(self.sess.run(ar_layer1_w))
            print(self.sess.run(ar_layer2_mean_w))

            print(self.sess.run(ar_mean, feed_dict=feed_dict))
            asdfasdf
