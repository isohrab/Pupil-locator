import tensorflow as tf
from base_model import BaseModel
from tensorflow.contrib.layers import l2_regularizer

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
                                 kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
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
                             kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                             name="conv27")

        x = tf.nn.leaky_relu(x, alpha=0.1, name="ReLu")
        self.logger.log("conv {}: {}".format("Logits", x.get_shape()))

        # Logits
        self.logits = tf.reshape(x, shape=(-1, self.cfg["output_dim"]), name='y')

        self.loss = tf.losses.mean_squared_error(self.Y,
                                                 self.logits,
                                                 weights=[self.cfg["output_weights"][0:self.cfg["output_dim"]]])



