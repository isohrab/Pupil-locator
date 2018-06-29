import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer, l2_regularizer


class GAP(object):
    """
    Convolution model:
    """

    def __init__(self, model_name, cfg, logger):
        super(GAP, self).__init__(model_name, cfg, logger)
        self.logger.log("building the model...")
        self.init_placeholders()
        self.init_forward()
        self.init_optimizer()
        self.summary_op = tf.summary.merge_all()

    def init_forward(self):
        k = 4
        cnn_input = self.X

        assert len(self.cfg["filter_sizes"]) == len(self.cfg["n_filters"])

        for i in range(len(self.cfg["filter_sizes"])):
            # cnn_input = tf.nn.dropout(cnn_input, self.keep_prob)
            cnn_input = tf.layers.conv2d(cnn_input,
                                         filters=self.cfg["n_filters"][i] * k,
                                         kernel_size=self.cfg["filter_sizes"][i],
                                         padding='same',
                                         activation=tf.nn.leaky_relu,
                                         kernel_initializer=xavier_initializer(),
                                         kernel_regularizer=l2_regularizer(self.cfg["l2_beta"]))

            cnn_input = tf.layers.batch_normalization(cnn_input,
                                                      training=self.train_flag)
            # print what happen to layers! :)
            self.logger.log("layer {} conv2d: {}".format(i, cnn_input.get_shape()))

            if self.cfg["max_pool"][i] == 1:
                cnn_input = tf.layers.max_pooling2d(cnn_input, pool_size=2, strides=2)
                # print what happen to layers! :)
                self.logger.log("layer {} MaxPool: {}".format(i, cnn_input.get_shape()))

        _, w, h, _ = cnn_input.get_shape()
        cnn_input = tf.layers.average_pooling2d(cnn_input, (w, h), strides=1)
        self.logger.log("layer {} AvgPool: {}".format(i, cnn_input.get_shape()))

        # Define fully connected layer
        # First we need to reshape cnn output to [batch_size, -1]
        a = tf.contrib.layers.flatten(cnn_input)
        h_prev = a.get_shape().as_list()[1]
        for i, h in enumerate(self.cfg["fc_layers"]):
            # by using fully_connected, tf will take care of X*W+b
            with tf.name_scope("fc_layer" + str(i)):
                with tf.name_scope("weight_" + str(i)):
                    initial_value = tf.truncated_normal([h_prev, h], stddev=0.001)
                    w = tf.Variable(initial_value, name="fc_w_" + str(i))
                    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, w)
                    self.variable_summaries(w)

                with tf.name_scope("bias_" + str(i)):
                    b = tf.Variable(tf.zeros([h]), name='fc_b_' + str(i))
                    self.variable_summaries(b)

                with tf.name_scope("Wx_plus_b_" + str(i)):
                    z = tf.matmul(a, w) + b

                with tf.name_scope("L_ReLu_" + str(i)):
                    a = tf.nn.leaky_relu(z)

            h_prev = h

            # show fully connected layers shape
            self.logger.log("layer {} fully connected: {}".format(i, a.get_shape()))

        self.logits = tf.contrib.layers.fully_connected(a, self.cfg["output_dim"], activation_fn=None)
        # self.logits = tf.reshape(cnn_input, shape=(-1, self.cfg["output_dim"]))

        self.loss = tf.losses.mean_squared_error(self.Y,
                                                 self.logits,
                                                 weights=[self.cfg["output_weights"][0:self.cfg["output_dim"]]])

        # Training summary for the current batch_loss
        tf.summary.scalar('loss', self.loss)
