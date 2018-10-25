import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.contrib.layers import xavier_initializer_conv2d, l2_regularizer

from base_model import BaseModel


class Simple(object):
    """
    Convolution model:
    """

    def __init__(self, model_name, cfg, logger):
        super(Simple, self).__init__(model_name, cfg, logger)
        self.logger.log("building the model...")
        self.init_placeholders()
        self.init_forward()
        self.init_optimizer()
        self.summary_op = tf.summary.merge_all()

    def init_forward(self):
        cnn_input = self.X
        xavi = tf.contrib.layers.xavier_initializer_conv2d()
        assert len(self.cfg["filter_sizes"]) == len(self.cfg["n_filters"])

        for i in range(len(self.cfg["filter_sizes"])):
            cnn_input = tf.nn.dropout(cnn_input, self.keep_prob)
            cnn_input = tf.layers.conv2d(cnn_input,
                                         filters=self.cfg["n_filters"][i],
                                         kernel_size=self.cfg["filter_sizes"][i],
                                         padding='same',
                                         activation=tf.nn.leaky_relu,
                                         kernel_initializer=xavi)

            cnn_input = tf.layers.batch_normalization(cnn_input,
                                                      training=self.train_flag,
                                                      momentum=0.99,
                                                      epsilon=0.001,
                                                      center=True,
                                                      scale=True)
            if self.cfg["max_pool"][i] == 1:
                cnn_input = tf.layers.max_pooling2d(cnn_input, pool_size=2, strides=2)

            # print what happen to layers! :)
            self.logger.log("layer {} conv2d: {}".format(i, cnn_input.get_shape()))

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
                    self.variable_summaries(w)

                with tf.name_scope("bias_" + str(i)):
                    b = tf.Variable(tf.zeros([h]), name='fc_b_' + str(i))
                    self.variable_summaries(b)

                with tf.name_scope("Wx_plus_b_" + str(i)):
                    z = tf.matmul(a, w) + b

                with tf.name_scope("L_ReLu_" + str(i)):
                    a = tf.nn.leaky_relu(z)

            h_prev = h
            # fc_input = tf.contrib.layers.fully_connected(fc_input, h, activation_fn=tf.nn.leaky_relu)

            # use batch normalization. With batch normalization we can get 1% better results
            # fc_input = tf.layers.batch_normalization(fc_input, training=(self.mode == "train"))

            # use dropout
            # fc_input = tf.nn.dropout(fc_input, keep_prob=self.keep_prob)

            # show fully connected layers shape
            self.logger.log("layer {} fully connected: {}".format(i, a.get_shape()))

        self.logits = tf.contrib.layers.fully_connected(a, self.cfg["output_dim"], activation_fn=None)
        # self.logits = tf.reshape(cnn_input, shape=(-1, self.cfg["output_dim"]))

        self.loss = tf.losses.mean_squared_error(self.Y,
                                                 self.logits,
                                                 weights=[self.cfg["output_weights"][0:self.cfg["output_dim"]]])

        # Training summary for the current batch_loss
        tf.summary.scalar('loss', self.loss)


# YOLO implementation
# https://github.com/WojciechMormul/yolo2/blob/master/train.py
class YOLO(BaseModel):
    """
    Convolution model: Yolo
    """

    def __init__(self, model_name, cfg, logger):
        super(YOLO, self).__init__(model_name, cfg, logger)
        self.logger.log("building the model...")
        self.init_placeholders()
        self.init_forward()
        self.init_optimizer()
        self.summary_op = tf.summary.merge_all()

    def maxpool_layer(self, x, size, stride, name):
        with tf.name_scope(name):
            x = tf.layers.max_pooling2d(x, size, stride, padding='SAME')

        return x

    def conv_layer(self, x, kernel, depth, train_logical, name):
        with tf.variable_scope(name):
            # x = tf.nn.dropout(x, keep_prob=self.keep_prob)
            x = tf.layers.conv2d(x, depth, kernel, padding='SAME',
                                 use_bias=False,
                                 kernel_initializer=xavier_initializer_conv2d(),
                                 kernel_regularizer=l2_regularizer(self.cfg["l2_beta"]))

            x = tf.layers.batch_normalization(x, training=train_logical)

            x = tf.nn.leaky_relu(x, alpha=0.1, name="ReLu")
        return x

    def passthrough_layer(self, a, b, kernel, depth, size, train_logical, name):
        b = self.conv_layer(b, kernel, depth, train_logical, name)
        b = tf.space_to_depth(b, size)
        y = tf.concat([a, b], axis=3)

        return y

    def init_forward(self):
        x = self.X
        self.logger.log("input shape: {}".format(x.get_shape()))

        i = 1

        # block 1
        x = self.conv_layer(x, (3, 3), i * 16, self.train_flag, 'conv1')
        self.logger.log("conv {}: {}".format(1, x.get_shape()))

        x = self.maxpool_layer(x, (2, 2), (2, 2), 'maxpool1')
        self.logger.log("maxpool {}: {}".format(1, x.get_shape()))

        x = self.conv_layer(x, (3, 3), i * 32, self.train_flag, 'conv2')
        self.logger.log("conv {}: {}".format(1, x.get_shape()))

        x = self.maxpool_layer(x, (2, 2), (2, 2), 'maxpool2')
        self.logger.log("maxpool {}: {}".format(2, x.get_shape()))

        # block 2
        x = self.conv_layer(x, (3, 3), i * 64, self.train_flag, 'conv3')
        self.logger.log("conv {}: {}".format(3, x.get_shape()))

        x = self.conv_layer(x, (1, 1), i * 32, self.train_flag, 'conv4')
        self.logger.log("conv {}: {}".format(4, x.get_shape()))

        x = self.conv_layer(x, (3, 3), i * 64, self.train_flag, 'conv5')
        self.logger.log("conv {}: {}".format(5, x.get_shape()))

        x = self.maxpool_layer(x, (2, 2), (2, 2), 'maxpool5')
        self.logger.log("maxpool {}: {}".format(5, x.get_shape()))

        # block 3
        x = self.conv_layer(x, (3, 3), i * 128, self.train_flag, 'conv6')
        self.logger.log("conv {}: {}".format(6, x.get_shape()))

        x = self.conv_layer(x, (1, 1), i * 64, self.train_flag, 'conv7')
        self.logger.log("conv {}: {}".format(7, x.get_shape()))

        x = self.conv_layer(x, (3, 3), i * 128, self.train_flag, 'conv8')
        self.logger.log("conv {}: {}".format(8, x.get_shape()))

        x = self.maxpool_layer(x, (2, 2), (2, 2), 'maxpool8')
        self.logger.log("maxpool {}: {}".format(8, x.get_shape()))

        # block 4
        x = self.conv_layer(x, (3, 3), i * 256, self.train_flag, 'conv9')
        self.logger.log("conv {}: {}".format(9, x.get_shape()))

        x = self.conv_layer(x, (1, 1), i * 128, self.train_flag, 'conv10')
        self.logger.log("conv {}: {}".format(10, x.get_shape()))

        x = self.conv_layer(x, (3, 3), i * 256, self.train_flag, 'conv11')
        self.logger.log("conv {}: {}".format(11, x.get_shape()))

        x = self.conv_layer(x, (1, 1), i * 128, self.train_flag, 'conv12')
        self.logger.log("conv {}: {}".format(12, x.get_shape()))

        passthrough = self.conv_layer(x, (3, 3), i * 256, self.train_flag, 'conv13')
        self.logger.log("conv {}: {}".format(13, x.get_shape()))

        x = self.maxpool_layer(passthrough, (2, 2), (2, 2), 'maxpool13')
        self.logger.log("maxpool {}: {}".format(13, x.get_shape()))

        # block 5
        x = self.conv_layer(x, (3, 3), i * 512, self.train_flag, 'conv14')
        self.logger.log("conv {}: {}".format(14, x.get_shape()))

        x = self.conv_layer(x, (1, 1), i * 256, self.train_flag, 'conv15')
        self.logger.log("conv {}: {}".format(15, x.get_shape()))

        x = self.conv_layer(x, (3, 3), i * 512, self.train_flag, 'conv16')
        self.logger.log("conv {}: {}".format(16, x.get_shape()))

        x = self.conv_layer(x, (1, 1), i * 256, self.train_flag, 'conv17')
        self.logger.log("conv {}: {}".format(17, x.get_shape()))

        x = self.conv_layer(x, (3, 3), i * 512, self.train_flag, 'conv18')
        self.logger.log("conv {}: {}".format(18, x.get_shape()))

        # block 6
        x = self.conv_layer(x, (3, 3), i * 512, self.train_flag, 'conv19')
        self.logger.log("conv {}: {}".format(19, x.get_shape()))

        x = self.conv_layer(x, (3, 3), i * 512, self.train_flag, 'conv20')
        self.logger.log("conv {}: {}".format(20, x.get_shape()))

        x = self.passthrough_layer(x, passthrough, (3, 3), i * 32, 2, self.train_flag, 'conv21')
        self.logger.log("conv {}: {}".format(21, x.get_shape()))

        x = self.conv_layer(x, (3, 3), i * 512, self.train_flag, 'conv22')
        self.logger.log("conv {}: {}".format(22, x.get_shape()))

        x = self.maxpool_layer(x, (2, 2), (2, 2), 'maxpool22')
        self.logger.log("maxpool {}: {}".format(22, x.get_shape()))

        x = self.conv_layer(x, (3, 3), i * 512, self.train_flag, 'conv23')
        self.logger.log("conv {}: {}".format(23, x.get_shape()))

        x = self.maxpool_layer(x, (2, 2), (2, 2), 'maxpool23')
        self.logger.log("maxpool {}: {}".format(23, x.get_shape()))

        x = self.conv_layer(x, (3, 3), i * 512, self.train_flag, 'conv24')
        self.logger.log("conv {}: {}".format(24, x.get_shape()))

        x = self.maxpool_layer(x, (2, 2), (2, 2), 'maxpool24')
        self.logger.log("maxpool {}: {}".format(24, x.get_shape()))

        x = self.conv_layer(x, (3, 3), i * 512, self.train_flag, 'conv26')
        self.logger.log("conv {}: {}".format(26, x.get_shape()))

        # Final layer
        # x = self.conv_layer(x, (1, 1), self.cfg["output_dim"], self.train_flag, 'conv27')
        x = tf.layers.conv2d(x, self.cfg["output_dim"], (1, 1),
                             padding='SAME',
                             use_bias=False,
                             kernel_initializer=xavier_initializer_conv2d(),
                             name="conv27")

        x = tf.nn.leaky_relu(x, alpha=0.1, name="ReLu")
        self.logger.log("conv {}: {}".format("Logits", x.get_shape()))

        # Logits
        self.logits = tf.reshape(x, shape=(-1, self.cfg["output_dim"]), name='y')

        self.loss = tf.losses.mean_squared_error(self.Y,
                                                 self.logits,
                                                 weights=[self.cfg["output_weights"][0:self.cfg["output_dim"]]])


class NASNET(BaseModel):
    """
    Convolution model:
    """

    def __init__(self, model_name, cfg, logger):
        super(NASNET, self).__init__(model_name, cfg, logger)
        self.logger.log("building the model...")
        self.init_placeholders()
        self.init_forward()
        self.init_optimizer()
        self.summary_op = tf.summary.merge_all()

    def maxpool_layer(self, x, size, stride, name):
        with tf.name_scope(name):
            x = tf.layers.max_pooling2d(x, size, stride, padding='SAME')

        return x

    def conv_layer(self, x, kernel, depth, train_logical, name):
        with tf.variable_scope(name):
            # x = tf.nn.dropout(x, keep_prob=self.keep_prob)
            x = tf.layers.conv2d(x, depth, kernel, padding='SAME',
                                 use_bias=False,
                                 kernel_initializer=xavier_initializer_conv2d(),
                                 kernel_regularizer=l2_regularizer(self.cfg["l2_beta"]))

            x = tf.layers.batch_normalization(x, training=train_logical)

            x = tf.nn.leaky_relu(x, alpha=0.1, name="ReLu")
        return x

    def init_forward(self):
        module = hub.Module("https://tfhub.dev/google/imagenet/mobilenet_v2_100_192/feature_vector/1",
                            trainable=True,
                            name="NASNET")
        module.ModuleSpec.get_tags()

        a = module(self.X)

        h_prev = a.get_shape().as_list()[1]
        layers = [512, 128]
        for i, h in enumerate(layers):
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

                with tf.name_scope("Batch_norm_" + str(i)):
                    z_bn = tf.layers.batch_normalization(z, training=self.train_flag)

                with tf.name_scope("L_ReLu_" + str(i)):
                    a = tf.nn.leaky_relu(z_bn)

            h_prev = h
            self.logger.log("layer {} fully connected: {}".format(i, a.get_shape()))

        self.logits = tf.contrib.layers.fully_connected(a, self.cfg["output_dim"], activation_fn=None)

        self.loss = tf.losses.mean_squared_error(self.Y, self.logits)

        # Training summary for the current batch_loss
        tf.summary.scalar('loss', self.loss)


class Inception(BaseModel):
    """
    Google inception model
    """

    def __init__(self, model_name, cfg, logger):
        super(Inception, self).__init__(model_name, cfg, logger)
        self.m = 0.5
        self.l2_reg = l2_regularizer(cfg["l2_beta"])
        self.logger.log("building the model...")
        self.init_placeholders()
        self.init_forward()
        self.init_optimizer()
        self.summary_op = tf.summary.merge_all()

    def bn_lrelu(self, x, train_logical):
        x = tf.layers.batch_normalization(x, training=train_logical, momentum=0.9997, scale=True, center=True)
        x = tf.nn.leaky_relu(x, alpha=0.17)
        return x

    # Inception Block A
    def block_a(self, net, name_scope, is_training):
        with tf.variable_scope(name_or_scope=name_scope,
                               default_name="Inception_block_A"):
            # Branch 0, 1x1
            with tf.variable_scope("branch_0"):
                branch_0 = tf.layers.conv2d(inputs=net,
                                            filters=96 * self.m,
                                            kernel_size=(1, 1),
                                            strides=(1, 1),
                                            padding='SAME',
                                            kernel_regularizer=self.l2_reg,
                                            kernel_initializer=xavier_initializer_conv2d(),
                                            name="conv2d_0a_1x1")

                branch_0 = self.bn_lrelu(branch_0, is_training)

            # Branch 1: 1x1 + 3x3
            with tf.variable_scope("branch_1"):
                branch_1 = tf.layers.conv2d(inputs=net,
                                            filters=64 * self.m,
                                            kernel_size=(1, 1),
                                            strides=(1, 1),
                                            padding='SAME',
                                            kernel_regularizer=self.l2_reg,
                                            kernel_initializer=xavier_initializer_conv2d(),
                                            name="conv2d_1a_1x1")
                branch_1 = self.bn_lrelu(branch_1, is_training)

                branch_1 = tf.layers.conv2d(inputs=branch_1,
                                            filters=96 * self.m,
                                            kernel_size=(3, 3),
                                            strides=(1, 1),
                                            padding='SAME',
                                            kernel_regularizer=self.l2_reg,
                                            kernel_initializer=xavier_initializer_conv2d(),
                                            name="conv2d_1b_3x3")
                branch_1 = self.bn_lrelu(branch_1, is_training)

            # Branch 2: 1x1 + 3x3 + 3x3
            with tf.variable_scope("branch_2"):
                branch_2 = tf.layers.conv2d(inputs=net,
                                            filters=64 * self.m,
                                            kernel_size=(1, 1),
                                            strides=(1, 1),
                                            padding='SAME',
                                            kernel_regularizer=self.l2_reg,
                                            kernel_initializer=xavier_initializer_conv2d(),
                                            name="conv2d_2a_1x1")
                branch_2 = self.bn_lrelu(branch_2, is_training)

                branch_2 = tf.layers.conv2d(inputs=branch_2,
                                            filters=96 * self.m,
                                            kernel_size=(3, 3),
                                            strides=(1, 1),
                                            padding='SAME',
                                            kernel_regularizer=self.l2_reg,
                                            kernel_initializer=xavier_initializer_conv2d(),
                                            name="conv2d_2b_3x3")
                branch_2 = self.bn_lrelu(branch_2, is_training)

                branch_2 = tf.layers.conv2d(inputs=branch_2,
                                            filters=96 * self.m,
                                            kernel_size=(3, 3),
                                            strides=(1, 1),
                                            padding='SAME',
                                            kernel_regularizer=self.l2_reg,
                                            kernel_initializer=xavier_initializer_conv2d(),
                                            name="conv2d_2c_3x3")
                branch_2 = self.bn_lrelu(branch_2, is_training)

            # Branch 3: AvgPool + 1x1
            with tf.variable_scope("branch_3"):
                branch_3 = tf.layers.average_pooling2d(inputs=net,
                                                       pool_size=(3, 3),
                                                       strides=(1, 1),
                                                       padding='SAME',
                                                       name="AvgPool_3a_3x3")

                branch_3 = tf.layers.conv2d(inputs=branch_3,
                                            filters=96 * self.m,
                                            kernel_size=(1, 1),
                                            strides=(1, 1),
                                            padding='SAME',
                                            kernel_regularizer=self.l2_reg,
                                            kernel_initializer=xavier_initializer_conv2d(),
                                            name="conv2d_3b_1x1")
                branch_3 = self.bn_lrelu(branch_3, is_training)

            return tf.concat([branch_0, branch_1, branch_2, branch_3], axis=3)

    # Reduction block A
    def block_a_reduction(self, net, name_scope, is_training):
        with tf.variable_scope(name_or_scope=name_scope,
                               default_name="Reduction_block_A"):
            # Branch 0, 3x3(V2)
            with tf.variable_scope("branch_0"):
                branch_0 = tf.layers.conv2d(inputs=net,
                                            filters=384 * self.m,
                                            kernel_size=(3, 3),
                                            strides=(2, 2),
                                            padding='VALID',
                                            kernel_regularizer=self.l2_reg,
                                            kernel_initializer=xavier_initializer_conv2d(),
                                            name="conv2d_0a_3x3V2")

                branch_0 = self.bn_lrelu(branch_0, is_training)

            # Branch 1: 1x1 + 3x3 + 3x3V2
            with tf.variable_scope("branch_1"):
                branch_1 = tf.layers.conv2d(inputs=net,
                                            filters=192 * self.m,
                                            kernel_size=(1, 1),
                                            strides=(1, 1),
                                            padding='SAME',
                                            kernel_regularizer=self.l2_reg,
                                            kernel_initializer=xavier_initializer_conv2d(),
                                            name="conv2d_1a_1x1")

                branch_1 = self.bn_lrelu(branch_1, is_training)

                branch_1 = tf.layers.conv2d(inputs=branch_1,
                                            filters=224 * self.m,
                                            kernel_size=(3, 3),
                                            strides=(1, 1),
                                            padding="SAME",
                                            kernel_regularizer=self.l2_reg,
                                            kernel_initializer=xavier_initializer_conv2d(),
                                            name="conv2_1b_3x3")
                branch_1 = self.bn_lrelu(branch_1, is_training)

                branch_1 = tf.layers.conv2d(inputs=branch_1,
                                            filters=256 * self.m,
                                            kernel_size=(3, 3),
                                            strides=(2, 2),
                                            padding="VALID",
                                            kernel_regularizer=self.l2_reg,
                                            kernel_initializer=xavier_initializer_conv2d(),
                                            name="conv2_1c_3x3V2")
                branch_1 = self.bn_lrelu(branch_1, is_training)

            # Branch 2: MaxPool(3x3)
            with tf.variable_scope("branch_3"):
                branch_2 = tf.layers.max_pooling2d(inputs=net,
                                                   pool_size=(3, 3),
                                                   strides=(2, 2),
                                                   padding='VALID',
                                                   name="MaxPool_2a_3x3V2")

        return tf.concat([branch_0, branch_1, branch_2], axis=3)

    # Inception Block B
    def block_b(self, net, name_scope, is_training):
        with tf.variable_scope(name_or_scope=name_scope,
                               default_name="Inception_block_B"):
            # Branch 0: 1x1
            with tf.variable_scope("branch_0"):
                branch_0 = tf.layers.conv2d(inputs=net,
                                            filters=384 * self.m,
                                            kernel_size=(1, 1),
                                            strides=(1, 1),
                                            padding="SAME",
                                            kernel_regularizer=self.l2_reg,
                                            kernel_initializer=xavier_initializer_conv2d(),
                                            name="Conv2d_0a_1x1")
                branch_0 = self.bn_lrelu(branch_0, is_training)

            # branch 1: 1x1 + 1x7 + 7x1
            with tf.variable_scope("branch_1"):
                branch_1 = tf.layers.conv2d(inputs=net,
                                            filters=192 * self.m,
                                            kernel_size=(1, 1),
                                            strides=(1, 1),
                                            padding="SAME",
                                            kernel_regularizer=self.l2_reg,
                                            kernel_initializer=xavier_initializer_conv2d(),
                                            name="Conv2d_1a_1x1")
                branch_1 = self.bn_lrelu(branch_1, is_training)

                branch_1 = tf.layers.conv2d(inputs=branch_1,
                                            filters=224 * self.m,
                                            kernel_size=(1, 7),
                                            strides=(1, 1),
                                            padding="SAME",
                                            kernel_regularizer=self.l2_reg,
                                            kernel_initializer=xavier_initializer_conv2d(),
                                            name="Conv2d_1b_1x7")
                branch_1 = self.bn_lrelu(branch_1, is_training)

                branch_1 = tf.layers.conv2d(inputs=branch_1,
                                            filters=256 * self.m,
                                            kernel_size=(7, 1),
                                            strides=(1, 1),
                                            padding="SAME",
                                            kernel_regularizer=self.l2_reg,
                                            kernel_initializer=xavier_initializer_conv2d(),
                                            name="Conv2d_1c_7x1")
                branch_1 = self.bn_lrelu(branch_1, is_training)

            # branch 2: 1x1 + 1x7 + 7x1 + 1x7 + 7x1
            with tf.variable_scope("branch_2"):
                branch_2 = tf.layers.conv2d(inputs=net,
                                            filters=192 * self.m,
                                            kernel_size=(1, 1),
                                            strides=(1, 1),
                                            padding="SAME",
                                            kernel_regularizer=self.l2_reg,
                                            kernel_initializer=xavier_initializer_conv2d(),
                                            name="Conv2_2a_1x1")
                branch_2 = self.bn_lrelu(branch_2, is_training)

                branch_2 = tf.layers.conv2d(inputs=branch_2,
                                            filters=192 * self.m,
                                            kernel_size=(1, 7),
                                            strides=(1, 1),
                                            padding="SAME",
                                            kernel_regularizer=self.l2_reg,
                                            kernel_initializer=xavier_initializer_conv2d(),
                                            name="Conv2d_2b_1x7")
                branch_2 = self.bn_lrelu(branch_2, is_training)

                branch_2 = tf.layers.conv2d(inputs=branch_2,
                                            filters=224 * self.m,
                                            kernel_size=(7, 1),
                                            strides=(1, 1),
                                            padding="SAME",
                                            kernel_regularizer=self.l2_reg,
                                            kernel_initializer=xavier_initializer_conv2d(),
                                            name="Conv2d_2c_7x1")
                branch_2 = self.bn_lrelu(branch_2, is_training)

                branch_2 = tf.layers.conv2d(inputs=branch_2,
                                            filters=224 * self.m,
                                            kernel_size=(1, 7),
                                            strides=(1, 1),
                                            padding="SAME",
                                            kernel_regularizer=self.l2_reg,
                                            kernel_initializer=xavier_initializer_conv2d(),
                                            name="Conv2d_2d_1x7")
                branch_2 = self.bn_lrelu(branch_2, is_training)

                branch_2 = tf.layers.conv2d(inputs=branch_2,
                                            filters=256 * self.m,
                                            kernel_size=(7, 1),
                                            strides=(1, 1),
                                            padding="SAME",
                                            kernel_regularizer=self.l2_reg,
                                            kernel_initializer=xavier_initializer_conv2d(),
                                            name="Conv2d_2e_7x1")
                branch_2 = self.bn_lrelu(branch_2, is_training)

            # Branch 3: AvgPool + 1x1
            with tf.variable_scope("branch_3"):
                branch_3 = tf.layers.average_pooling2d(inputs=net,
                                                       pool_size=(3, 3),
                                                       strides=(1, 1),
                                                       padding="SAME",
                                                       name="AvgPool_3a_3x3")

                branch_3 = tf.layers.conv2d(inputs=branch_3,
                                            filters=128 * self.m,
                                            kernel_size=(1, 1),
                                            strides=(1, 1),
                                            padding="SAME",
                                            kernel_regularizer=self.l2_reg,
                                            kernel_initializer=xavier_initializer_conv2d(),
                                            name="conv2d_3b_1x1")
                branch_3 = self.bn_lrelu(branch_3, is_training)

        return tf.concat([branch_0, branch_1, branch_2, branch_3], axis=3)

    # Reduction block B
    def block_b_reduction(self, net, name_scope, is_training):
        with tf.variable_scope(name_or_scope=name_scope,
                               default_name="Reduction_block_B"):
            # Branch 0: 1x1 + 3x3(V,2)
            with tf.variable_scope("branch_0"):
                branch_0 = tf.layers.conv2d(inputs=net,
                                            filters=192 * self.m,
                                            kernel_size=(1, 1),
                                            strides=(1, 1),
                                            padding="SAME",
                                            kernel_regularizer=self.l2_reg,
                                            kernel_initializer=xavier_initializer_conv2d(),
                                            name="conv2d_0a_1x1")
                branch_0 = self.bn_lrelu(branch_0, is_training)

                branch_0 = tf.layers.conv2d(inputs=branch_0,
                                            filters=192 * self.m,
                                            kernel_size=(3, 3),
                                            strides=(2, 2),
                                            padding="VALID",
                                            kernel_regularizer=self.l2_reg,
                                            kernel_initializer=xavier_initializer_conv2d(),
                                            name="conv2d_0b_3x3V2")
                branch_0 = self.bn_lrelu(branch_0, is_training)

            # Branch 1: 1x1 + 1x7 + 7x1 + 3x3(V,2)
            with tf.variable_scope("branch_1"):
                branch_1 = tf.layers.conv2d(inputs=net,
                                            filters=256 * self.m,
                                            kernel_size=(1, 1),
                                            strides=(1, 1),
                                            padding="SAME",
                                            kernel_regularizer=self.l2_reg,
                                            kernel_initializer=xavier_initializer_conv2d(),
                                            name="Conv2d_1a_1x1")
                branch_1 = self.bn_lrelu(branch_1, is_training)

                branch_1 = tf.layers.conv2d(inputs=branch_1,
                                            filters=256 * self.m,
                                            kernel_size=(1, 7),
                                            strides=(1, 1),
                                            padding="SAME",
                                            kernel_regularizer=self.l2_reg,
                                            kernel_initializer=xavier_initializer_conv2d(),
                                            name="conv2d_1b_1x7")
                branch_1 = self.bn_lrelu(branch_1, is_training)

                branch_1 = tf.layers.conv2d(inputs=branch_1,
                                            filters=320 * self.m,
                                            kernel_size=(7, 1),
                                            strides=(1, 1),
                                            padding="SAME",
                                            kernel_regularizer=self.l2_reg,
                                            kernel_initializer=xavier_initializer_conv2d(),
                                            name="conv2d_1c_7x1")
                branch_1 = self.bn_lrelu(branch_1, is_training)

                branch_1 = tf.layers.conv2d(inputs=branch_1,
                                            filters=320 * self.m,
                                            kernel_size=(3, 3),
                                            strides=(2, 2),
                                            padding="VALID",
                                            kernel_regularizer=self.l2_reg,
                                            kernel_initializer=xavier_initializer_conv2d(),
                                            name="Conv2d_1d_3x3V2")
                branch_1 = self.bn_lrelu(branch_1, is_training)

            # Branch 2: MaxPool 3x3 (V,2)
            with tf.variable_scope("branch_2"):
                branch_2 = tf.layers.max_pooling2d(inputs=net,
                                                   pool_size=(3, 3),
                                                   strides=(2, 2),
                                                   padding="VALID",
                                                   name="MaxPool_2a_3x3V2")

        return tf.concat([branch_0, branch_1, branch_2], axis=3)

    # Inception Block C
    def block_c(self, net, name_scope, is_training):
        with tf.variable_scope(name_or_scope=name_scope,
                               default_name="Inception_Block_C"):
            # Branch 0: 1x1
            with tf.variable_scope("branch_0"):
                branch_0 = tf.layers.conv2d(inputs=net,
                                            filters=256 * self.m,
                                            kernel_size=(1, 1),
                                            strides=(1, 1),
                                            padding="SAME",
                                            kernel_regularizer=self.l2_reg,
                                            kernel_initializer=xavier_initializer_conv2d(),
                                            name="Conv2d_0a_1x1")
                branch_0 = self.bn_lrelu(branch_0, is_training)

            # Branch 1: 1x1 {1x3, 3x1}
            with tf.variable_scope("branch_1"):
                branch_1 = tf.layers.conv2d(inputs=net,
                                            filters=384 * self.m,
                                            kernel_size=(1, 1),
                                            strides=(1, 1),
                                            padding="SAME",
                                            kernel_regularizer=self.l2_reg,
                                            kernel_initializer=xavier_initializer_conv2d(),
                                            name="conv2d_1a_1x1")
                branch_1 = self.bn_lrelu(branch_1, is_training)

                branch_1a = tf.layers.conv2d(inputs=branch_1,
                                             filters=256 * self.m,
                                             kernel_size=(1, 3),
                                             strides=(1, 1),
                                             padding="SAME",
                                             kernel_regularizer=self.l2_reg,
                                             kernel_initializer=xavier_initializer_conv2d(),
                                             name="conv2d_1b0_1x3")
                branch_1a = self.bn_lrelu(branch_1a, is_training)

                branch_1b = tf.layers.conv2d(inputs=branch_1,
                                             filters=256 * self.m,
                                             kernel_size=(3, 1),
                                             strides=(1, 1),
                                             padding="SAME",
                                             kernel_regularizer=self.l2_reg,
                                             kernel_initializer=xavier_initializer_conv2d(),
                                             name="conv2d_1b1_3x1")
                branch_1b = self.bn_lrelu(branch_1b, is_training)

                branch_1 = tf.concat([branch_1a, branch_1b], axis=3)

            # Branch 2: 1x1, 3x1, 1x3 {3x1, 1x3}
            with tf.variable_scope("branch_2"):
                branch_2 = tf.layers.conv2d(inputs=net,
                                            filters=384 * self.m,
                                            kernel_size=(1, 1),
                                            strides=(1, 1),
                                            padding="SAME",
                                            kernel_regularizer=self.l2_reg,
                                            kernel_initializer=xavier_initializer_conv2d(),
                                            name="conv2d_2a_1x1")
                branch_2 = self.bn_lrelu(branch_2, is_training)

                branch_2 = tf.layers.conv2d(inputs=branch_2,
                                            filters=448 * self.m,
                                            kernel_size=(1, 3),
                                            strides=(1, 1),
                                            padding="SAME",
                                            kernel_regularizer=self.l2_reg,
                                            kernel_initializer=xavier_initializer_conv2d(),
                                            name="conv2d_2b_1x3")
                branch_2 = self.bn_lrelu(branch_2, is_training)

                branch_2 = tf.layers.conv2d(inputs=branch_2,
                                            filters=512 * self.m,
                                            kernel_size=(3, 1),
                                            strides=(1, 1),
                                            padding="SAME",
                                            kernel_regularizer=self.l2_reg,
                                            kernel_initializer=xavier_initializer_conv2d(),
                                            name="conv2d_2c_3x1")
                branch_2 = self.bn_lrelu(branch_2, is_training)

                branch_2a = tf.layers.conv2d(inputs=branch_2,
                                             filters=256 * self.m,
                                             kernel_size=(1, 3),
                                             strides=(1, 1),
                                             padding="SAME",
                                             kernel_regularizer=self.l2_reg,
                                             kernel_initializer=xavier_initializer_conv2d(),
                                             name="conv2d_2d0_1x3")
                branch_2a = self.bn_lrelu(branch_2a, is_training)

                branch_2b = tf.layers.conv2d(inputs=branch_2,
                                             filters=256 * self.m,
                                             kernel_size=(3, 1),
                                             strides=(1, 1),
                                             padding="SAME",
                                             kernel_regularizer=self.l2_reg,
                                             kernel_initializer=xavier_initializer_conv2d(),
                                             name="conv2d_2d1_3x1")
                branch_2b = self.bn_lrelu(branch_2b, is_training)

                branch_2 = tf.concat([branch_2a, branch_2b], axis=3)

            # Branch 3: AvgPool, 1x1
            with tf.variable_scope("branch_3"):
                branch_3 = tf.layers.average_pooling2d(inputs=net,
                                                       pool_size=(3, 3),
                                                       strides=(1, 1),
                                                       padding="SAME",
                                                       name="AvgPool_3a_3x3")
                branch_3 = tf.layers.conv2d(inputs=branch_3,
                                            filters=256 * self.m,
                                            kernel_size=(1, 1),
                                            padding="SAME",
                                            kernel_regularizer=self.l2_reg,
                                            kernel_initializer=xavier_initializer_conv2d(),
                                            name="Conv2d_3b_1x1")
                branch_3 = self.bn_lrelu(branch_3, is_training)

        return tf.concat([branch_0, branch_1, branch_2, branch_3], axis=3)

    def init_forward(self):
        # make the stem
        net = self.X
        self.logger.log("net shape {}".format(net.get_shape()))

        # Begin Inception Model
        with tf.variable_scope(name_or_scope="InceptionV4"):
            net = tf.layers.conv2d(inputs=net,
                                   filters=32 * self.m,
                                   kernel_size=(3, 3),
                                   strides=(2, 2),
                                   padding="VALID",
                                   kernel_regularizer=self.l2_reg,
                                   kernel_initializer=xavier_initializer_conv2d(),
                                   name="conv2d_stem0_3x3V2")
            net = self.bn_lrelu(net, self.train_flag)
            self.logger.log("stem0 shape {}".format(net.get_shape()))

            net = tf.layers.conv2d(inputs=net,
                                   filters=32 * self.m,
                                   kernel_size=(3, 3),
                                   strides=(1, 1),
                                   padding="VALID",
                                   kernel_regularizer=self.l2_reg,
                                   kernel_initializer=xavier_initializer_conv2d(),
                                   name="conv2d_stem1_3x3V1")
            net = self.bn_lrelu(net, self.train_flag)
            self.logger.log("stem1 shape {}".format(net.get_shape()))

            net = tf.layers.conv2d(inputs=net,
                                   filters=64 * self.m,
                                   kernel_size=(3, 3),
                                   strides=(1, 1),
                                   padding="SAME",
                                   kernel_regularizer=self.l2_reg,
                                   kernel_initializer=xavier_initializer_conv2d(),
                                   name="Conv2d_stem2_3x3")
            net = self.bn_lrelu(net, self.train_flag)
            self.logger.log("stem2 shape {}".format(net.get_shape()))

            with tf.variable_scope("Mixed_3a"):
                with tf.variable_scope("branch_0"):
                    net_a = tf.layers.conv2d(inputs=net,
                                             filters=96 * self.m,
                                             kernel_size=(3, 3),
                                             strides=(2, 2),
                                             padding="VALID",
                                             kernel_regularizer=self.l2_reg,
                                             kernel_initializer=xavier_initializer_conv2d(),
                                             name="Conv2d_0a_3x3s2")
                    net_a = self.bn_lrelu(net_a, self.train_flag)

                with tf.variable_scope("branch_1"):
                    net_b = tf.layers.max_pooling2d(inputs=net,
                                                    pool_size=(3, 3),
                                                    strides=(2, 2),
                                                    padding="VALID",
                                                    name="MaxPool_1a_3x3s2")

            net = tf.concat([net_a, net_b], axis=3)
            self.logger.log("Mixed_3a shape {}".format(net.get_shape()))

            with tf.variable_scope("mixed_4a"):
                # Branch 0: 1x1, 7x1, 1x7, 3x3v
                with tf.variable_scope("branch_0"):
                    branch_0 = tf.layers.conv2d(inputs=net,
                                                filters=64 * self.m,
                                                kernel_size=(1, 1),
                                                strides=(1, 1),
                                                padding="SAME",
                                                kernel_regularizer=self.l2_reg,
                                                kernel_initializer=xavier_initializer_conv2d(),
                                                name="Conv2d_0a_3x3")
                    branch_0 = self.bn_lrelu(branch_0, self.train_flag)

                    branch_0 = tf.layers.conv2d(inputs=branch_0,
                                                filters=64 * self.m,
                                                kernel_size=(7, 1),
                                                strides=(1, 1),
                                                padding="SAME",
                                                kernel_regularizer=self.l2_reg,
                                                kernel_initializer=xavier_initializer_conv2d(),
                                                name="Conv2d_0b_7x1")
                    branch_0 = self.bn_lrelu(branch_0, self.train_flag)

                    branch_0 = tf.layers.conv2d(inputs=branch_0,
                                                filters=64 * self.m,
                                                kernel_size=(1, 7),
                                                strides=(1, 1),
                                                padding="SAME",
                                                kernel_regularizer=self.l2_reg,
                                                kernel_initializer=xavier_initializer_conv2d(),
                                                name="Conv2d_0c_1x7")
                    branch_0 = self.bn_lrelu(branch_0, self.train_flag)

                    branch_0 = tf.layers.conv2d(inputs=branch_0,
                                                filters=96 * self.m,
                                                kernel_size=(3, 3),
                                                strides=(1, 1),
                                                padding="VALID",
                                                kernel_regularizer=self.l2_reg,
                                                kernel_initializer=xavier_initializer_conv2d(),
                                                name="Conv2d_0d_3x3V")
                    branch_0 = self.bn_lrelu(branch_0, self.train_flag)

                # Branch 1: 1x1, 3x3v
                with tf.variable_scope("branch_1"):
                    branch_1 = tf.layers.conv2d(inputs=net,
                                                filters=64 * self.m,
                                                kernel_size=(1, 1),
                                                strides=(1, 1),
                                                padding="SAME",
                                                kernel_regularizer=self.l2_reg,
                                                kernel_initializer=xavier_initializer_conv2d(),
                                                name="Conv2d_0a_3x3")
                    branch_1 = self.bn_lrelu(branch_1, self.train_flag)

                    branch_1 = tf.layers.conv2d(inputs=branch_1,
                                                filters=96 * self.m,
                                                kernel_size=(3, 3),
                                                strides=(1, 1),
                                                padding="VALID",
                                                kernel_regularizer=self.l2_reg,
                                                kernel_initializer=xavier_initializer_conv2d(),
                                                name="Conv2d_0b_3x3V")
                    branch_1 = self.bn_lrelu(branch_1, self.train_flag)

            net = tf.concat([branch_0, branch_1], axis=3)
            self.logger.log("mixed_4a shape {}".format(net.get_shape()))

            with tf.variable_scope("Mixed_5a"):
                # Branch 0: 3x3
                with tf.variable_scope("branch_0"):
                    branch_0 = tf.layers.conv2d(inputs=net,
                                                filters=192 * self.m,
                                                kernel_size=(3, 3),
                                                strides=(2, 2),
                                                padding="VALID",
                                                kernel_regularizer=self.l2_reg,
                                                kernel_initializer=xavier_initializer_conv2d(),
                                                name="Conv2d_0a_3x3v")
                    branch_0 = self.bn_lrelu(branch_0, self.train_flag)

                # Branch 1: MaxPool 3x3s2
                with tf.variable_scope("branch_1"):
                    branch_1 = tf.layers.max_pooling2d(inputs=net,
                                                       pool_size=(3, 3),
                                                       strides=(2, 2),
                                                       padding="VALID",
                                                       name="MaxPool_0a_3x3s2")

            net = tf.concat([branch_0, branch_1], axis=3)
            self.logger.log("Mixed_5a shape {}".format(net.get_shape()))

            # Block A: 3x
            net = self.block_a(net, "Block_A0", self.train_flag)
            self.logger.log("Block_A0 shape {}".format(net.get_shape()))

            net = self.block_a(net, "Block_A1", self.train_flag)
            self.logger.log("Block_A1 shape {}".format(net.get_shape()))

            net = self.block_a(net, "Block_A2", self.train_flag)
            self.logger.log("Block_A2 shape {}".format(net.get_shape()))

            # Block A: Reduction
            net = self.block_a_reduction(net, "Reduction_A", self.train_flag)
            self.logger.log("Reduction_A shape {}".format(net.get_shape()))


            # Block B: 4x
            net = self.block_b(net, "Block_B0", self.train_flag)
            self.logger.log("Block_B0 shape {}".format(net.get_shape()))

            net = self.block_b(net, "Block_B1", self.train_flag)
            self.logger.log("Block_B1 shape {}".format(net.get_shape()))

            net = self.block_b(net, "Block_B2", self.train_flag)
            self.logger.log("Block_B2 shape {}".format(net.get_shape()))

            net = self.block_b(net, "Block_B3", self.train_flag)
            self.logger.log("Block_B3 shape {}".format(net.get_shape()))

            # # Block B reducttion
            # net = self.block_b_reduction(net, "Reduction_B", self.train_flag)
            # self.logger.log("Reduction_B shape {}".format(net.get_shape()))
            #
            # # Block C: 4x
            # net = self.block_c(net, "Block_C0", self.train_flag)
            # self.logger.log("Block_C0 shape {}".format(net.get_shape()))
            #
            # net = self.block_c(net, "Block_C1", self.train_flag)
            # self.logger.log("Block_C1 shape {}".format(net.get_shape()))
            #
            # net = self.block_c(net, "Block_C2", self.train_flag)
            # self.logger.log("Block_C1 shape {}".format(net.get_shape()))
            #
            # net = self.block_c(net, "Block_C3", self.train_flag)
            # self.logger.log("Block_C1 shape {}".format(net.get_shape()))

            net = tf.nn.dropout(net, self.keep_prob, name="net_dropout")

            self.GAP = tf.reduce_mean(net, axis=[1, 2], name="GAP")
            self.logger.log("GAP shape {}".format(self.GAP.get_shape()))

            # Final layer
            units = self.GAP.get_shape().as_list()[1]
            net = tf.reshape(self.GAP, (-1, 1, 1, units), name="reshaping")
            net = tf.layers.conv2d(net, self.cfg["output_dim"], (1, 1),
                                   padding='VALID',
                                   kernel_initializer=xavier_initializer_conv2d(),
                                   kernel_regularizer=self.l2_reg,
                                   use_bias=False,
                                   name="final_conv")

            net = tf.nn.relu(net, name="logits_relu")
            self.logger.log("Final layer {}: {}".format("Logits", net.get_shape()))

            # Logits
            self.logits = tf.reshape(net, shape=(-1, self.cfg["output_dim"]), name='y')

            self.loss = tf.losses.huber_loss(labels=self.Y,
                                             predictions=self.logits,
                                             weights=[self.cfg["output_weights"][0:self.cfg["output_dim"]]],
                                             delta=1.0)

            # Training summary for the current batch_loss
            tf.summary.scalar('loss', self.loss)


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
                                         kernel_initializer=xavier_initializer_conv2d(),
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
