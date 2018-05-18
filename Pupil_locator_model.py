import tensorflow as tf


class Model(object):
    """
    Convolution model:
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.mode = 'train'
        # self.logger = logger
        self.max_gradient_norm = cfg["MAX_GRADIANT_NORM"]
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.global_epoch_step = tf.Variable(0, trainable=False, name='global_epoch_step')
        self.global_epoch_step_op = tf.assign(self.global_epoch_step, self.global_epoch_step + 1)
        self.build_model()

    def build_model(self):
        print("building the model...")

        self.init_placeholders()
        self.init_layers()
        self.summary_op = tf.summary.merge_all()

    def init_placeholders(self):
        # encoder inputs are include </s> tokens. e.g: "hello world </s>". So we can use them as decoder_output too.
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

    def init_layers(self):
        cnn_input = self.X
        xavi = tf.contrib.layers.xavier_initializer_conv2d()
        assert len(self.cfg["filter_sizes"]) == len(self.cfg["n_filters"])

        for i in range(len(self.cfg["filter_sizes"])):
            cnn_input = tf.layers.conv2d(cnn_input,
                                         filters=self.cfg["n_filters"][i],
                                         kernel_size=self.cfg["filter_sizes"][i],
                                         padding='same',
                                         activation=tf.nn.leaky_relu,
                                         kernel_initializer=xavi)

            # max pooling only on layers which are set True
            if self.cfg["max_pool"][i] == 1:
                stride = 2
                if self.cfg["filter_sizes"][i] == 1:
                    stride = 1
                cnn_input = tf.layers.max_pooling2d(cnn_input, pool_size=2, strides=stride)

            # print what happen to layers! :)
            print("layer {} conv2d: {}".format(i, cnn_input.get_shape()))

        # ## Define fully connected layer
        # # First we need to reshape cnn output to [batch_size, -1]
        # fc_input = tf.contrib.layers.flatten(cnn_input)
        #
        # for h in self.cfg["fc_layers"]:
        #     # by using fully_connected, tf will take care of X*W+b
        #     fc_input = tf.contrib.layers.fully_connected(fc_input, h)
        #
        #     # use batch normalization. With batch normalization we can get 1% better results
        #     # fc_input = tf.layers.batch_normalization(fc_input, training=(self.mode == "train"))
        #
        #     # use dropout
        #     # fc_input = tf.nn.dropout(fc_input, keep_prob=self.keep_prob)

        # self.logits = tf.contrib.layers.fully_connected(cnn_input, self.cfg["output_dim"], activation_fn=None)
        self.logits = tf.reshape(cnn_input, shape=(-1, self.cfg["output_dim"]))
        self.loss = tf.losses.mean_squared_error(self.Y, self.logits)

        # Training summary for the current batch_loss
        tf.summary.scalar('loss', self.loss)

        # Construct graphs for minimizing loss
        self.init_optimizer()

    def init_optimizer(self):
        print("setting optimizer..")
        # trainable_params = tf.trainable_variables()

        learning_rate = tf.train.exponential_decay(self.cfg["learning_rate"],
                                                   self.global_step,
                                                   self.cfg["decay_step"],
                                                   self.cfg["decay_rate"],
                                                   staircase=True)
        # TODO: need to handle all optimization
        self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.cfg["learning_rate"]).minimize(self.loss, global_step=self.global_step)
        # # Compute gradients of loss w.r.t. all trainable variables
        # gradients = tf.gradients(self.loss, trainable_params)
        #
        # # Clip gradients by a given maximum_gradient_norm
        # clip_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
        #
        # # Update the model
        # self.updates = self.opt.apply_gradients(zip(clip_gradients, trainable_params),
        #                                         global_step=self.global_step)

    def train(self, sess, images, labels, keep_prob):
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
                      self.keep_prob: keep_prob}

        output_feed = [self.opt,  # Update Op that does optimization
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
                      self.keep_prob: 1.0}

        output_feed = [self.loss,  # Loss for current batch
                       self.summary_op,
                       self.logits]

        outputs = sess.run(output_feed, input_feed)
        return outputs[0], outputs[1], outputs[2]

    def predict(self, sess, images):
        self.mode = 'test'
        # Input feeds for dropout
        input_feed = {self.X.name: images,
                      self.keep_prob.name: 1.0}

        output_feed = [self.logits]
        outputs = sess.run(output_feed, input_feed)

        return outputs[0]
