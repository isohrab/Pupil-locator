import tensorflow as tf


# YOLO implementation
# https://github.com/WojciechMormul/yolo2/blob/master/train.py
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
        self.init_yolo()
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

        self.train_flag = tf.placeholder(dtype=tf.bool, name='flag_placeholder')

    def maxpool_layer(self, x, size, stride, name):
        with tf.name_scope(name):
            x = tf.layers.max_pooling2d(x, size, stride, padding='SAME')

        return x

    def conv_layer(self, x, kernel, depth, train_logical, name):

        with tf.variable_scope(name):
            x = tf.layers.conv2d(x, depth, kernel, padding='SAME',
                                 kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 bias_initializer=tf.zeros_initializer())
            x = tf.layers.batch_normalization(x, training=train_logical, momentum=0.99, epsilon=0.001, center=True,
                                              scale=True)
            x = tf.nn.dropout(x, self.keep_prob)

        return x

    def passthrough_layer(self, a, b, kernel, depth, size, train_logical, name):

        b = self.conv_layer(b, kernel, depth, train_logical, name)
        b = tf.space_to_depth(b, size)
        y = tf.concat([a, b], axis=3)

        return y


    def init_yolo(self):

        x = self.X
        self.logger.log("input shape: {}".format(x.get_shape()))

        # block 1
        x = self.conv_layer(x, (3, 3), 32, self.train_flag, 'conv1')
        self.logger.log("conv {}: {}".format(1, x.get_shape()))

        x = self.maxpool_layer(x, (2, 2), (2, 2), 'maxpool1')
        self.logger.log("maxpool {}: {}".format(1, x.get_shape()))

        x = self.conv_layer(x, (3, 3), 64, self.train_flag, 'conv2')
        self.logger.log("conv {}: {}".format(1, x.get_shape()))

        x = self.maxpool_layer(x, (2, 2), (2, 2), 'maxpool2')
        self.logger.log("maxpool {}: {}".format(2, x.get_shape()))

        # block 2
        x = self.conv_layer(x, (3, 3), 128, self.train_flag, 'conv3')
        self.logger.log("conv {}: {}".format(3, x.get_shape()))

        x = self.conv_layer(x, (1, 1), 64, self.train_flag, 'conv4')
        self.logger.log("conv {}: {}".format(4, x.get_shape()))

        x = self.conv_layer(x, (3, 3), 128, self.train_flag, 'conv5')
        self.logger.log("conv {}: {}".format(5, x.get_shape()))

        x = self.maxpool_layer(x, (2, 2), (2, 2), 'maxpool5')
        self.logger.log("maxpool {}: {}".format(5, x.get_shape()))

        # block 3
        x = self.conv_layer(x, (3, 3), 256, self.train_flag, 'conv6')
        self.logger.log("conv {}: {}".format(6, x.get_shape()))

        x = self.conv_layer(x, (1, 1), 128, self.train_flag, 'conv7')
        self.logger.log("conv {}: {}".format(7, x.get_shape()))

        x = self.conv_layer(x, (3, 3), 256, self.train_flag, 'conv8')
        self.logger.log("conv {}: {}".format(8, x.get_shape()))

        x = self.maxpool_layer(x, (2, 2), (2, 2), 'maxpool8')
        self.logger.log("maxpool {}: {}".format(8, x.get_shape()))

        # block 4
        x = self.conv_layer(x, (3, 3), 512, self.train_flag, 'conv9')
        self.logger.log("conv {}: {}".format(9, x.get_shape()))

        x = self.conv_layer(x, (1, 1), 256, self.train_flag, 'conv10')
        self.logger.log("conv {}: {}".format(10, x.get_shape()))

        x = self.conv_layer(x, (3, 3), 512, self.train_flag, 'conv11')
        self.logger.log("conv {}: {}".format(11, x.get_shape()))

        x = self.conv_layer(x, (1, 1), 256, self.train_flag, 'conv12')
        self.logger.log("conv {}: {}".format(12, x.get_shape()))

        passthrough = self.conv_layer(x, (3, 3), 512, self.train_flag, 'conv13')
        self.logger.log("conv {}: {}".format(13, x.get_shape()))

        x = self.maxpool_layer(passthrough, (2, 2), (2, 2), 'maxpool13')
        self.logger.log("maxpool {}: {}".format(13, x.get_shape()))

        # block 5
        x = self.conv_layer(x, (3, 3), 1024, self.train_flag, 'conv14')
        self.logger.log("conv {}: {}".format(14, x.get_shape()))

        x = self.conv_layer(x, (1, 1), 512, self.train_flag, 'conv15')
        self.logger.log("conv {}: {}".format(15, x.get_shape()))

        x = self.conv_layer(x, (3, 3), 1024, self.train_flag, 'conv16')
        self.logger.log("conv {}: {}".format(16, x.get_shape()))

        x = self.conv_layer(x, (1, 1), 512, self.train_flag, 'conv17')
        self.logger.log("conv {}: {}".format(17, x.get_shape()))

        x = self.conv_layer(x, (3, 3), 1024, self.train_flag, 'conv18')
        self.logger.log("conv {}: {}".format(18, x.get_shape()))

        # block 6
        x = self.conv_layer(x, (3, 3), 1024, self.train_flag, 'conv19')
        self.logger.log("conv {}: {}".format(19, x.get_shape()))

        x = self.conv_layer(x, (3, 3), 1024, self.train_flag, 'conv20')
        self.logger.log("conv {}: {}".format(20, x.get_shape()))

        x = self.passthrough_layer(x, passthrough, (3, 3), 64, 2, self.train_flag, 'conv21')
        self.logger.log("conv {}: {}".format(21, x.get_shape()))

        x = self.conv_layer(x, (3, 3), 1024, self.train_flag, 'conv22')
        self.logger.log("conv {}: {}".format(22, x.get_shape()))

        x = self.maxpool_layer(x, (2, 2), (2, 2), 'maxpool22')
        self.logger.log("maxpool {}: {}".format(22, x.get_shape()))

        x = self.conv_layer(x, (3, 3), 1024, self.train_flag, 'conv23')
        self.logger.log("conv {}: {}".format(23, x.get_shape()))

        x = self.maxpool_layer(x, (2, 2), (2, 2), 'maxpool23')
        self.logger.log("maxpool {}: {}".format(23, x.get_shape()))

        x = self.conv_layer(x, (3, 3), 1024, self.train_flag, 'conv24')
        self.logger.log("conv {}: {}".format(24, x.get_shape()))

        x = self.maxpool_layer(x, (2, 2), (2, 2), 'maxpool24')
        self.logger.log("maxpool {}: {}".format(24, x.get_shape()))

        x = self.conv_layer(x, (3, 3), 1024, self.train_flag, 'conv26')
        self.logger.log("conv {}: {}".format(26, x.get_shape()))

        x = self.conv_layer(x, (1, 1), self.cfg["output_dim"], self.train_flag, 'conv27')
        self.logger.log("conv {}: {}".format(23, x.get_shape()))

        # Logits
        self.logits = tf.reshape(x, shape=(-1, 5), name='y')
        self.loss = tf.losses.mean_squared_error(self.Y,
                                                 self.logits,
                                                 weights=[[3.0, 3.0, 1.0, 1.0]])

        # Training summary for the current batch_loss
        tf.summary.scalar('loss', self.loss)

        # Construct graphs for minimizing loss
        self.init_optimizer()

    def init_optimizer(self):
        print("setting optimizer..")
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            trainable_params = tf.trainable_variables()

            learning_rate = tf.train.exponential_decay(self.cfg["learning_rate"],
                                                       self.global_step,
                                                       self.cfg["decay_step"],
                                                       self.cfg["decay_rate"],
                                                       staircase=True)
            # TODO: need to handle all optimization
            # self.opt = tf.train.AdamOptimizer(learning_rate=self.cfg["learning_rate"]).minimize(self.loss,
            #                                                                         global_step=self.global_step)
            self.opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
            # Compute gradients of loss w.r.t. all trainable variables
            gradients = tf.gradients(self.loss, trainable_params)

            # Clip gradients by a given maximum_gradient_norm
            clip_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)

            # Update the model
            self.update = self.opt.apply_gradients(zip(clip_gradients, trainable_params),
                                                   global_step=self.global_step)

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
                      self.keep_prob.name: keep_prob,
                      self.train_flag.name: True}

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
                      self.keep_prob.name: 1.0}

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
