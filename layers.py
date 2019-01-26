import tensorflow as tf


def lrelu(x, leak=0.2, name="lrelu", alt_relu_impl=False):

    with tf.variable_scope(name):
        if alt_relu_impl:
            f1 = 0.5 * (1 + leak)
            f2 = 0.5 * (1 - leak)
            return f1 * x + f2 * abs(x)
        else:
            return tf.maximum(x, leak * x)


def instance_norm(x):
    with tf.variable_scope("instance_norm"):
        out = tf.contrib.layers.instance_norm(x)

        return out

def batch_norm(x, is_training = True):

    with tf.variable_scope("batch_norm"):
        return tf.contrib.layers.batch_norm(x, is_training=is_training, decay=0.90, scale=True, center=True,
                                            variables_collections=["internal_batchnorm_variables"],
                                            updates_collections=None)


def general_conv2d(inputconv, o_d=64, f_h=7, f_w=7, s_h=1, s_w=1, stddev=0.01,
                   padding="VALID", name="conv2d", do_norm=True, do_relu=True, keep_rate=None,
                   relufactor=0, norm_type=None, is_training=True):
    with tf.variable_scope(name):

        conv = tf.contrib.layers.conv2d(
            inputconv, o_d, f_w, s_w, padding,
            activation_fn=None,
            weights_initializer=tf.truncated_normal_initializer(
                stddev=stddev
            ),
            biases_initializer=None
        )
        if not keep_rate is None:
            conv = tf.nn.dropout(conv, keep_rate)

        if do_norm:
            if norm_type is None:
                print "normalization type is not specified!"
                quit()
            elif norm_type=='Ins':
                conv = instance_norm(conv)
            elif norm_type=='Batch':
                conv = batch_norm(conv, is_training)

        if do_relu:
            if(relufactor == 0):
                conv = tf.nn.relu(conv, "relu")
            else:
                conv = lrelu(conv, relufactor, "lrelu")

        return conv


def general_conv2d_ga(inputconv, o_d=64, f_h=7, f_w=7, s_h=1, s_w=1, stddev=0.02,
                   padding="VALID", name="conv2d", do_norm=True, do_relu=True, keep_rate=None,
                   relufactor=0, norm_type=None, is_training=True):
    with tf.variable_scope(name):

        conv = tf.contrib.layers.conv2d(
            inputconv, o_d, f_w, s_w, padding,
            activation_fn=None,
            weights_initializer=tf.truncated_normal_initializer(
                stddev=stddev
            ),
            biases_initializer=tf.constant_initializer(0.0)
        )
        if not keep_rate is None:
            conv = tf.nn.dropout(conv, keep_rate)

        if do_norm:
            if norm_type is None:
                print "normalization type is not specified!"
                quit()
            elif norm_type=='Ins':
                conv = instance_norm(conv)
            elif norm_type=='Batch':
                conv = batch_norm(conv, is_training)

        if do_relu:
            if(relufactor == 0):
                conv = tf.nn.relu(conv, "relu")
            else:
                conv = lrelu(conv, relufactor, "lrelu")

        return conv


def dilate_conv2d(inputconv, i_d=64, o_d=64, f_h=7, f_w=7, rate=2, stddev=0.01,
                   padding="VALID", name="dilate_conv2d", do_norm=True, do_relu=True, keep_rate=None,
                   relufactor=0, norm_type=None, is_training=True):
    with tf.variable_scope(name):
        f_1 = tf.get_variable('weights', [f_h, f_w, i_d, o_d], initializer=tf.truncated_normal_initializer(stddev=stddev))
        b_1 = tf.get_variable('biases', [o_d], initializer=tf.constant_initializer(0.0, tf.float32))
        di_conv_2d = tf.nn.atrous_conv2d(inputconv, f_1, rate=rate, padding=padding)

        if not keep_rate is None:
            di_conv_2d = tf.nn.dropout(di_conv_2d, keep_rate)

        if do_norm:
            if norm_type is None:
                print "normalization type is not specified!"
                quit()
            elif norm_type=='Ins':
                di_conv_2d = instance_norm(di_conv_2d)
            elif norm_type=='Batch':
                di_conv_2d = batch_norm(di_conv_2d, is_training)

        if do_relu:
            if(relufactor == 0):
                di_conv_2d = tf.nn.relu(di_conv_2d, "relu")
            else:
                di_conv_2d = lrelu(di_conv_2d, relufactor, "lrelu")

        return di_conv_2d


def general_deconv2d(inputconv, outshape, o_d=64, f_h=7, f_w=7, s_h=1, s_w=1,
                     stddev=0.02, padding="VALID", name="deconv2d",
                     do_norm=True, do_relu=True, relufactor=0, norm_type=None, is_training=True):
    with tf.variable_scope(name):

        conv = tf.contrib.layers.conv2d_transpose(
            inputconv, o_d, [f_h, f_w],
            [s_h, s_w], padding,
            activation_fn=None,
            weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
            biases_initializer=tf.constant_initializer(0.0)
        )

        if do_norm:
            if norm_type is None:
                print "normalization type is not specified!"
                quit()
            elif norm_type=='Ins':
                conv = instance_norm(conv)
            elif norm_type=='Batch':
                conv = batch_norm(conv, is_training)

        if do_relu:
            if(relufactor == 0):
                conv = tf.nn.relu(conv, "relu")
            else:
                conv = lrelu(conv, relufactor, "lrelu")

        return conv
