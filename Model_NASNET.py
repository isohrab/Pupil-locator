import tensorflow as tf
import tensorflow_hub as hub

class Model(object):
    """
    Convolution model:
    """

    def __init__(self, model_name, cfg, logger):
        self.cfg = cfg
        self.model_name = model_name
        self.logger = logger
        self.model_dir = "models/" + model_name + "/"
        self.mode = 'train'
        self.max_gradient_norm = cfg["MAX_GRADIANT_NORM"]
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.global_epoch_step = tf.Variable(0, trainable=False, name='global_epoch_step')
        self.global_epoch_step_op = tf.assign(self.global_epoch_step, self.global_epoch_step + 1)
        self.build_model()

    def build_model(self):
        self.logger.log("building the model...")
        self.init_placeholders()
        self.train_nas()
        self.summary_op = tf.summary.merge_all()

    def init_placeholders(self):
        # shape: [Batch_size, Width, Height, Channels]
        self.X = tf.placeholder(dtype=tf.float32,
                                shape=(None,
                                       self.cfg["image_width"],
                                       self.cfg["image_height"],
                                       self.cfg["image_channel"]),
                                name="images_input")


        # shape: [Batch_size, 5] (x,y,w,h,a)
        self.Y = tf.placeholder(dtype=tf.float32,
                                shape=(None, self.cfg["output_dim"]),
                                name="ground_truth")

        self.keep_prob = tf.placeholder(dtype=tf.float32,
                                        shape=(),
                                        name="keep_prob")

        self.train_flag = tf.placeholder(dtype=tf.bool, name='flag_placeholder')

        self.learning_rate = tf.placeholder(dtype=tf.float32, shape=(), name="learning_rate")

    def maxpool_layer(self, x, size, stride, name):
        with tf.name_scope(name):
            x = tf.layers.max_pooling2d(x, size, stride, padding='SAME')

        return x

    def conv_layer(self, x, kernel, depth, train_logical, name):

        with tf.variable_scope(name):
            # x = tf.nn.dropout(x, keep_prob=self.keep_prob)
            x = tf.layers.conv2d(x, depth, kernel, padding='SAME',
                                 use_bias=False,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())

            x = tf.layers.batch_normalization(x, training=train_logical)

            x = tf.nn.leaky_relu(x, alpha=0.1, name="ReLu")
        return x

    def train_nas(self):
        module = hub.Module("https://tfhub.dev/google/imagenet/mobilenet_v2_100_192/feature_vector/1",
                            trainable=True,
                            name="NASNET")

        a = module(self.X)

        _, h = a.get_shape()
        x = tf.reshape(a, (-1, 1, 1, h))

        x = self.conv_layer(x, (1, 1), 1024, self.train_flag, 'conv1')
        self.logger.log("conv {}: {}".format(1, x.get_shape()))

        x = self.conv_layer(x, (1, 1), 1024, self.train_flag, 'conv2')
        self.logger.log("conv {}: {}".format(1, x.get_shape()))

        x = self.conv_layer(x, (1, 1), 512, self.train_flag, 'conv3')
        self.logger.log("conv {}: {}".format(1, x.get_shape()))

        x = self.conv_layer(x, (1, 1), 512, self.train_flag, 'conv4')
        self.logger.log("conv {}: {}".format(1, x.get_shape()))

        x = self.conv_layer(x, (1, 1), 256, self.train_flag, 'conv5')
        self.logger.log("conv {}: {}".format(1, x.get_shape()))

        x = self.conv_layer(x, (1, 1), 128, self.train_flag, 'conv6')
        self.logger.log("conv {}: {}".format(1, x.get_shape()))

        x = self.conv_layer(x, (1, 1), 64, self.train_flag, 'conv7')
        self.logger.log("conv {}: {}".format(1, x.get_shape()))

        x = self.conv_layer(x, (1, 1), 32, self.train_flag, 'conv8')
        self.logger.log("conv {}: {}".format(1, x.get_shape()))

        x = self.conv_layer(x, (1, 1), 16, self.train_flag, 'conv9')
        self.logger.log("conv {}: {}".format(1, x.get_shape()))

        x = self.conv_layer(x, (1, 1), 8, self.train_flag, 'conv10')
        self.logger.log("conv {}: {}".format(1, x.get_shape()))

        x = self.conv_layer(x, (1, 1), self.cfg["output_dim"], self.train_flag, 'conv11')
        self.logger.log("conv {}: {}".format(1, x.get_shape()))

        # Logits
        self.logits = tf.reshape(x, shape=(-1, self.cfg["output_dim"]), name='y')


        # h_prev = a.get_shape().as_list()[1]
        # layers = [512, 128]
        # for i, h in enumerate(layers):
        #     # by using fully_connected, tf will take care of X*W+b
        #     with tf.name_scope("fc_layer" + str(i)):
        #         with tf.name_scope("weight_" + str(i)):
        #             initial_value = tf.truncated_normal([h_prev, h], stddev=0.001)
        #             w = tf.Variable(initial_value, name="fc_w_" + str(i))
        #             self.variable_summaries(w)
        #
        #         with tf.name_scope("bias_" + str(i)):
        #             b = tf.Variable(tf.zeros([h]), name='fc_b_' + str(i))
        #             self.variable_summaries(b)
        #
        #         with tf.name_scope("Wx_plus_b_" + str(i)):
        #             z = tf.matmul(a, w) + b
        #
        #         with tf.name_scope("Batch_norm_" + str(i)):
        #             z_bn = tf.layers.batch_normalization(z,
        #                                                  momentum=0.95,
        #                                                  epsilon=1e-5,
        #                                                  training=self.train_flag)
        #         with tf.name_scope("L_ReLu_" + str(i)):
        #             a = tf.nn.leaky_relu(z_bn)
        #
        #     h_prev = h
        #     self.logger.log("layer {} fully connected: {}".format(i, a.get_shape()))

        # self.logits = tf.contrib.layers.fully_connected(a, self.cfg["output_dim"], activation_fn=None)

        self.loss = tf.losses.mean_squared_error(self.Y, self.logits)

        # Training summary for the current batch_loss
        tf.summary.scalar('loss', self.loss)

        # Construct graphs for minimizing loss
        self.init_optimizer()


    def init_optimizer(self):
        print("setting optimizer..")
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            trainable_params = tf.trainable_variables()

            self.opt = tf.train.AdamOptimizer(learning_rate=0.00001)

            # # add l2 loss
            # l2_loss = 0
            # for var in trainable_params:
            #     if var.name.find("weight") > 0 or var.name.find("kernel") > 0:
            #         l2_loss += 0.005 * tf.nn.l2_loss(var)

            # final_loss = self.loss + l2_loss
            # Compute gradients of loss w.r.t. all trainable variables
            gradients = tf.gradients(self.loss, trainable_params)

            # Clip gradients by a given maximum_gradient_norm
            clip_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)

            # Update the model
            self.update = self.opt.apply_gradients(zip(clip_gradients, trainable_params),
                                                   global_step=self.global_step)

    def train(self, sess, images, labels, keep_prob, lr):
        """Run a train step of the model feeding the given inputs.
        Args:
        session: tensorflow session to use.
        encoder_inputs: a numpy int matrix of [batch_size, max_source_time_steps]
            to feed as encoder inputs
        encoder_inputs_length: a numpy int vector of [batch_size]
            to feed as sequence lengths for each element in the given batch
        Returns:
            A triple consisting of gradient norm (or None if we did not do backward),
        average perplexity, and the outputs.
        """
        # Check if the model is 'training' mode
        self.mode = 'train'

        input_feed = {self.X.name: images,
                      self.Y.name: labels,
                      self.keep_prob.name: keep_prob,
                      self.train_flag.name: True,
                      self.learning_rate.name: lr}

        output_feed = [self.update,  # Update Op that does optimization
                       self.loss,  # Loss for current batch
                       self.summary_op]

        outputs = sess.run(output_feed, input_feed)
        return outputs[1], outputs[2]

    def eval(self, sess, images, labels):
        """Run a evaluation step of the model feeding the given inputs.
        Args:
        session: tensorflow session to use.
        encoder_inputs: a numpy int matrix of [batch_size, max_source_time_steps]
        to feed as encoder inputs
        encoder_inputs_length: a numpy int vector of [batch_size]
        to feed as sequence lengths for each element in the given batch
        Returns:
        A triple consisting of gradient norm (or None if we did not do backward),
        average perplexity, and the outputs.
        """
        self.mode = "eval"
        input_feed = {self.X.name: images,
                      self.Y.name: labels,
                      self.keep_prob.name: 1.0,
                      self.train_flag.name: False}

        output_feed = [self.loss,  # Loss for current batch
                       self.summary_op,
                       self.logits]

        outputs = sess.run(output_feed, input_feed)
        return outputs[0], outputs[1], outputs[2]

    def predict(self, sess, images):
        self.mode = 'test'
        # Input feeds for dropout
        input_feed = {self.X.name: images,
                      self.keep_prob.name: 1.0,
                      self.train_flag.name: False}

        output_feed = [self.logits]
        outputs = sess.run(output_feed, input_feed)

        return outputs[0]

    def restore(self, sess, path, var_list=None):
        # var_list = None returns the list of all saveable variables
        saver = tf.train.Saver(var_list)
        saver.restore(sess, save_path=path)
        self.logger.log('model restored from %s' % path)

    def variable_summaries(self, var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)
