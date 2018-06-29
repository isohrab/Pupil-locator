import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.contrib.layers import xavier_initializer, l2_regularizer
from base_model import BaseModel

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
                                 kernel_initializer=xavier_initializer,
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