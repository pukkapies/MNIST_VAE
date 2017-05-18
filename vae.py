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
from utils.training_utils import create_json
import json

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
    RESTORE_KEY = "to_restore_"
    DEBUG_KEY = 'debug_'

    def __init__(self, encoder, decoder, latent_dim, d_hyperparams={}, scope='VAE', save_graph_def=True,
                 model_name=None, log_dir="./log/", analysis_dir=None, model_to_restore=None):
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
        self.settings.update({'encoder_settings': encoder.settings,
                                             'decoder_settings': decoder.settings,
                                             'latent_dim': latent_dim,
                                             'scope': scope})
        self.settings.update(**d_hyperparams)

        if model_to_restore is None:
            print("Building new VAE model")
            self.encoder = encoder
            self.decoder = decoder
            self.latent_dim = latent_dim
            self.scope = scope

            self.model_name = '' if model_name is None else model_name
            model_datetime = datetime.now().strftime(r"%y%m%d_%H%M")
            self.model_folder = './training/saved_models/' + model_datetime + self.model_name + '/'
            if not os.path.exists(self.model_folder): os.makedirs(self.model_folder)

            self.settings_folder = self.model_folder + 'settings/'
            if not os.path.exists(self.settings_folder): os.makedirs(self.settings_folder)

            if analysis_dir is None:
                self.analysis_folder = self.model_folder + 'analysis/'
            else:
                self.analysis_folder = analysis_dir
            if not os.path.exists(self.analysis_folder): os.makedirs(self.analysis_folder)

            self._build_graph()
            self.sess.run(tf.global_variables_initializer())
        else:
            print('Restoring model: ', model_to_restore)
            self.model_folder = '/'.join((model_to_restore.split('/')[:-1])) + '/'
            self.settings_folder = self.model_folder + 'settings/'
            with open(self.settings_folder + 'settings.json') as json_file:
                self.settings = json.load(json_file)

            # Rebuild the graph
            meta_graph = os.path.abspath(model_to_restore)
            tf.train.import_meta_graph(meta_graph + ".meta").restore(self.sess, meta_graph)

            self.unpack_handles()

        if save_graph_def:  # tensorboard
            self.logger = tf.summary.FileWriter(log_dir, self.sess.graph)

    def unpack_handles(self):
        self.input_ph = self.sess.graph.get_collection(VAE.RESTORE_KEY + 'input_ph')[0]
        self.ar_mean = self.sess.graph.get_collection(VAE.RESTORE_KEY + 'ar_mean')[0]
        self.ar_logsigma = self.sess.graph.get_collection(VAE.RESTORE_KEY + 'ar_logsigma')[0]
        self.vae_output = self.sess.graph.get_collection(VAE.RESTORE_KEY + 'vae_output')[0]
        self.zT = self.sess.graph.get_collection(VAE.RESTORE_KEY + 'zT')[0]
        self.z_ = self.sess.graph.get_collection(VAE.RESTORE_KEY + 'z_')[0]
        self.x_reconstructed_ = self.sess.graph.get_collection(VAE.RESTORE_KEY + 'x_reconstructed')[0]
        self.cost = self.sess.graph.get_collection(VAE.RESTORE_KEY + 'cost')[0]
        self.train_op = self.sess.graph.get_collection(VAE.RESTORE_KEY + 'train_op')[0]
        self.rec_loss = self.sess.graph.get_collection(VAE.RESTORE_KEY + 'rec_loss')[0]
        self.kl_loss = self.sess.graph.get_collection(VAE.RESTORE_KEY + 'kl_loss')[0]
        self.global_step = self.sess.graph.get_collection(VAE.RESTORE_KEY + 'global_step')[0]
        return

    @property
    def step(self):
        """Train step"""
        return self.global_step.eval(session=self.sess)

    def save_model(self, outdir):
        """Saves the model if a self.saver object exists"""
        try:
            outfile = outdir + 'model'
            self.saver.save(self.sess, outfile, global_step=self.step)
            create_json(self.settings_folder + 'settings.json', self.settings)
        except AttributeError:
            print("Failed to save model at step {}".format(self.step))
            return

    def _build_graph(self):
        with tf.variable_scope(self.scope) as graph_scope:
            self.input_ph = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE])  # (batch_size, n_inputs)
            tf.add_to_collection(VAE.RESTORE_KEY + "input_ph", self.input_ph)
            print("input shape: ", self.input_ph.get_shape())

            self.global_step = tf.Variable(0, trainable=False, name="global_step")
            tf.add_to_collection(VAE.RESTORE_KEY + "global_step", self.global_step)

            prelatent_layer = self.encoder(self.input_ph)

            z0_mean = Dense(scope="z0_mean", size=self.latent_dim,
                           initializer=variance_scaling_initializer(scale=2.0, mode="fan_in", distribution="normal"))(prelatent_layer)
            z0_log_sigma = Dense(scope="z0_log_sigma", size=self.latent_dim,
                                initializer=variance_scaling_initializer(scale=2.0, mode="fan_in", distribution="normal"))(prelatent_layer)

            prior = DiagonalGaussian(tf.zeros(self.latent_dim), tf.ones(self.latent_dim))  # N(0, 1)
            posterior = DiagonalGaussian(z0_mean, z0_log_sigma)

            tf.add_to_collection(VAE.DEBUG_KEY + 'z0_mean', z0_mean)
            tf.add_to_collection(VAE.DEBUG_KEY + 'z0_log_sigma', z0_log_sigma)

            print("Finished setting up encoder")
            print([var._variable for var in tf.global_variables()])

            print('z0_mean shape: ', z0_mean.get_shape())
            print('z0_log_sigma shape: ', z0_log_sigma.get_shape())

            z0 = posterior.sample()
            print("Posterior sample shape: ", z0.get_shape())

            tf.add_to_collection(VAE.DEBUG_KEY + 'z0', z0)

            logqs = posterior.logprob(z0)

            # IAF Posterior
            # Create two AR layers
            first_AR_Dense = AR_Dense(8,
                                      variance_scaling_initializer(scale=2.0, mode="fan_avg", distribution="normal"),
                                      zerodiagonal=False, scope='ar_layer1', nonlinearity=tf.nn.elu)
            second_AR_Dense = AR_Dense(8,
                                      variance_scaling_initializer(scale=2.0, mode="fan_avg", distribution="normal"),
                                      zerodiagonal=False, scope='ar_layer2', nonlinearity=tf.nn.elu)
            AR_Dense_to_mean = AR_Dense(self.latent_dim,
                                               variance_scaling_initializer(scale=2.0, mode="fan_avg", distribution="normal"),
                                               zerodiagonal=True, scope='ar_layer_mean')
            AR_Dense_to_logsigma = AR_Dense(self.latent_dim,
                                                   variance_scaling_initializer(scale=2.0, mode="fan_avg", distribution="normal"),
                                                   zerodiagonal=True, scope='ar_layer_logsd')

            z1 = first_AR_Dense(z0)
            z2 = second_AR_Dense(z1)
            ar_mean = AR_Dense_to_mean(z2)
            ar_logsigma = AR_Dense_to_logsigma(z2)
            tf.add_to_collection(VAE.RESTORE_KEY + "ar_mean", ar_mean)
            tf.add_to_collection(VAE.RESTORE_KEY + "ar_logsigma", ar_logsigma)

            # self.ar_mean = ar_mean  # for debugging
            # self.ar_logsigma = ar_logsigma  # for debugging

            zT = (z0 - ar_mean) / tf.exp(ar_logsigma)
            tf.add_to_collection(VAE.RESTORE_KEY + 'zT', zT)
            print("Post AR z.shape", zT.get_shape())

            tf.add_to_collection(VAE.DEBUG_KEY + 'z1', z1)
            tf.add_to_collection(VAE.DEBUG_KEY + 'z2', z2)

            self.first_ar_layer_weights = first_AR_Dense.w  # for debugging
            self.second_ar_layer_weights_mean = AR_Dense_to_mean.w  # for debugging
            self.second_ar_layer_weights_logsigma = AR_Dense_to_logsigma.w  # for debugging

            tf.add_to_collection(VAE.DEBUG_KEY + 'ar_layer1_w', first_AR_Dense.w)
            tf.add_to_collection(VAE.DEBUG_KEY + 'ar_layer1_b', first_AR_Dense.b)
            tf.add_to_collection(VAE.DEBUG_KEY + 'ar_layer2_w', second_AR_Dense.w)
            tf.add_to_collection(VAE.DEBUG_KEY + 'ar_layer2_b', second_AR_Dense.b)

            tf.add_to_collection(VAE.DEBUG_KEY + 'ar_layer_mean_w', AR_Dense_to_mean.w)
            tf.add_to_collection(VAE.DEBUG_KEY + 'ar_layer_logsd_w', AR_Dense_to_logsigma.w)
            tf.add_to_collection(VAE.DEBUG_KEY + 'ar_layer_mean_b', AR_Dense_to_mean.b)
            tf.add_to_collection(VAE.DEBUG_KEY + 'ar_layer_logsd_b', AR_Dense_to_logsigma.b)

            logqs += ar_logsigma
            logps = prior.logprob(zT)
            kl_obj = logqs - logps  # (batch_size, latent_dim)

            kl_obj = tf.reduce_sum(kl_obj, [1])

            self.vae_output = self.decoder(zT)  # (batch_size, n_outputs)
            tf.add_to_collection(VAE.RESTORE_KEY + "vae_output", self.vae_output)

            print("vae output shape: ", self.vae_output.get_shape())

            print("Finished setting up decoder")
            print([var._variable for var in tf.global_variables()])

            ##########################################################################

            # Set up gradient calculation and optimizer
            # rec_loss = tf.losses.sigmoid_cross_entropy(input_ph, vae_output)
            rec_loss = self.crossEntropy(self.vae_output, self.input_ph)
            print('rec_loss shape:', rec_loss.get_shape())  # THINK I MIGHT NEED TO TAKE A MEAN HERE
            print('kl_obj shape:', kl_obj.get_shape())

            # # Kullback-Leibler divergence: mismatch b/w approximate vs. imposed/true posterior
            # kl_loss = self.kullback_leibler_diag_gaussian(z0_mean, z0_log_sigma)
            # print('kl_loss shape:', kl_loss.get_shape())

            with tf.name_scope("cost"):
                # average over minibatch
                self.cost = tf.reduce_mean(kl_obj + rec_loss, name="cost")
                tf.add_to_collection(VAE.RESTORE_KEY + "cost", self.cost)
                self.rec_loss = tf.reduce_mean(rec_loss, name='rec_loss')
                tf.add_to_collection(VAE.RESTORE_KEY + "rec_loss", self.rec_loss)
                self.kl_loss = tf.reduce_mean(kl_obj, name="kl_loss")
                tf.add_to_collection(VAE.RESTORE_KEY + "kl_loss", self.kl_loss)

            print("Defined loss functions")

            # optimization
            optimizer = tf.train.AdamOptimizer(learning_rate=self.settings['learning_rate'])
            tvars = tf.trainable_variables()
            grads_and_vars = optimizer.compute_gradients(self.cost, tvars)
            clipped = [(tf.clip_by_value(grad, -5, 5), tvar)  # gradient clipping
                       for grad, tvar in grads_and_vars]
            self.train_op = optimizer.apply_gradients(clipped, global_step=self.global_step, name="minimize_cost")
            tf.add_to_collection(VAE.RESTORE_KEY + "train_op", self.train_op)
            print("Defined training ops")

            print([var._variable for var in tf.global_variables()])

            # ops to directly explore latent space
            # defaults to prior z ~ N(0, I)
            with tf.name_scope("latent_in"):
                z_ = tf.placeholder_with_default(tf.random_normal([1, self.latent_dim]), shape=[1, self.latent_dim],
                                                 name="latent_in")
                tf.add_to_collection(VAE.RESTORE_KEY + "z_", z_)
            graph_scope.reuse_variables()  # No new variables should be created from this point on
            x_reconstructed_ = self.decoder(z_)
            tf.add_to_collection(VAE.RESTORE_KEY + "x_reconstructed", x_reconstructed_)

            return

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
        return self.sess.run(self.zT, feed_dict=feed_dict)

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
        self.best_cost = np.inf

        now = datetime.now().isoformat()[11:]
        print("------- Training begin: {} -------\n".format(now))

        while True:
            try:
                x = dataset.next_batch()  # (batch_size, n_inputs)

                feed_dict = {self.input_ph: x}
                
                fetches = [self.vae_output, self.cost, self.kl_loss, self.rec_loss, self.global_step, self.train_op]
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
                if cost < self.best_cost:
                    self.best_cost = cost
                    self.settings['best_cost'] = {'ELBO': float(cost),
                                                  'KL(q(z|x) || p(z))': float(kl_loss),
                                                  'reconstruction_loss': float(rec_loss)}

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
            x, labels = dataset.next_batch(images_only=False)  # (batch_size, n_inputs)
            feed_dict = {self.input_ph: x}

            fetches = [self.vae_output, self.cost, self.kl_loss, self.rec_loss, self.global_step]
            x_reconstructed, cost, kl_loss, rec_loss, i = self.sess.run(fetches, feed_dict=feed_dict)

            print(cost)

            z_mean = self.sess.graph.get_collection(VAE.DEBUG_KEY + 'z0_mean')[0]
            z_log_sigma = self.sess.graph.get_collection(VAE.DEBUG_KEY + 'z0_log_sigma')[0]
            z0 = self.sess.graph.get_collection(VAE.DEBUG_KEY + 'z0')[0]
            zT = self.sess.graph.get_collection(VAE.RESTORE_KEY + 'zT')[0]

            ar_mean = self.sess.graph.get_collection(VAE.RESTORE_KEY + 'ar_mean')[0]
            ar_logsigma = self.sess.graph.get_collection(VAE.RESTORE_KEY + 'ar_logsigma')[0]

            ar_layer1_w = self.sess.graph.get_collection(VAE.DEBUG_KEY + 'ar_layer1_w')[0]
            ar_layer1_b = self.sess.graph.get_collection(VAE.DEBUG_KEY + 'ar_layer1_b')[0]
            ar_layer2_w = self.sess.graph.get_collection(VAE.DEBUG_KEY + 'ar_layer2_w')[0]
            ar_layer2_b = self.sess.graph.get_collection(VAE.DEBUG_KEY + 'ar_layer2_b')[0]
            ar_layer_mean_w = self.sess.graph.get_collection(VAE.DEBUG_KEY + 'ar_layer_mean_w')[0]
            ar_layer_mean_b = self.sess.graph.get_collection(VAE.DEBUG_KEY + 'ar_layer_mean_b')[0]
            ar_layer_logsd_w = self.sess.graph.get_collection(VAE.DEBUG_KEY + 'ar_layer_logsd_w')[0]
            ar_layer_logsd_b = self.sess.graph.get_collection(VAE.DEBUG_KEY + 'ar_layer_logsd_b')[0]

            # Evaluate the above tensors
            z0_sample = self.sess.run(z0, feed_dict)
            zT_sample = self.sess.run(zT, feed_dict)  # Final Posterior approximation
            zT_2nd_sample = self.sess.run(zT, feed_dict)
            ar_mean_eval = self.sess.run(ar_mean, feed_dict)
            ar_logsigma_eval = self.sess.run(ar_logsigma, feed_dict)

            ar_layer1_w_eval = self.sess.run(ar_layer1_w)
            ar_layer1_b_eval = self.sess.run(ar_layer1_b)
            ar_layer_mean_w_eval = self.sess.run(ar_layer_mean_w)
            ar_layer_mean_b_eval = self.sess.run(ar_layer_mean_b)
            ar_layer_logsd_w_eval = self.sess.run(ar_layer_logsd_w)
            ar_layer_logsd_b_eval = self.sess.run(ar_layer_logsd_b)

            # print('ar_layer1_w:\n', ar_layer1_w_eval)
            # print('ar_layer1_b:\n', ar_layer1_b_eval)
            # print('ar_layer_mean_w:\n', ar_layer_mean_w_eval)
            # print('ar_layer_mean_b:\n', ar_layer_mean_b_eval)
            # print('ar_layer_logsd_w:\n', ar_layer_logsd_w_eval)
            # print('ar_layer_logsd_b:\n', ar_layer_logsd_b_eval)

            print("")
            print("************")
            print("")
            print("ar_mean:")
            print(ar_mean_eval)
            print("")
            print("ar_logsd:")
            print(ar_logsigma_eval)


            if self.settings['latent_dim'] == 2 and iterations==1:
                import matplotlib.pyplot as plt

                plt.figure()
                plt.title("z0 of VAE")
                colours = ['b', 'c', 'y', 'm', 'r', 'k', 'g', (0.2, 0.4, 0.6), (0.8, 0.3, 0.5), (0.1, 0.1, 0.5)]
                scatter_pts = []  # To store a list of np.arrays for each digit
                for i in range(10):
                    idx = (labels == i)
                    scatter_pts.append(plt.scatter(z0_sample[idx, 0], z0_sample[idx, 1], color=colours[i], alpha=0.8))
                plt.legend(tuple([scatter_object for scatter_object in scatter_pts]), (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
                           loc='upper right')

                plt.figure()
                plt.title("zT of VAE")
                colours = ['b', 'c', 'y', 'm', 'r', 'k', 'g', (0.2, 0.4, 0.6), (0.8, 0.3, 0.5), (0.1, 0.1, 0.5)]
                scatter_pts = []  # To store a list of np.arrays for each digit
                for i in range(10):
                    idx = (labels == i)
                    scatter_pts.append(plt.scatter(zT_sample[idx, 0], zT_sample[idx, 1], color=colours[i], alpha=0.8))
                plt.legend(tuple([scatter_object for scatter_object in scatter_pts]), (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
                           loc='upper right')

                plt.figure()
                plt.title("zT (2nd sample) of VAE")
                colours = ['b', 'c', 'y', 'm', 'r', 'k', 'g', (0.2, 0.4, 0.6), (0.8, 0.3, 0.5), (0.1, 0.1, 0.5)]
                scatter_pts = []  # To store a list of np.arrays for each digit
                for i in range(10):
                    idx = (labels == i)
                    scatter_pts.append(plt.scatter(zT_2nd_sample[idx, 0], zT_2nd_sample[idx, 1], color=colours[i], alpha=0.8))
                plt.legend(tuple([scatter_object for scatter_object in scatter_pts]), (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
                           loc='upper right')

                plt.show()
