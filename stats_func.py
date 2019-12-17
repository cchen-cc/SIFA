import tensorflow as tf
import numpy as np


def pixel_wise_softmax_2(output_map):
    exponential_map = tf.exp(output_map)
    sum_exp = tf.reduce_sum(exponential_map, 3, keep_dims=True)
    tensor_sum_exp = tf.tile(sum_exp, tf.stack([1, 1, 1, tf.shape(output_map)[3]]))
    return tf.clip_by_value(tf.div(exponential_map, tensor_sum_exp), -1.0 * 1e15, 1.0 * 1e15,
                            name="pixel_softmax_2d")


def jaccard(conf_matrix):
    num_cls = conf_matrix.shape[0]
    jac = np.zeros(num_cls)
    for ii in range(num_cls):
        pp = np.sum(conf_matrix[:, ii])
        gp = np.sum(conf_matrix[ii, :])
        hit = conf_matrix[ii, ii]
        jac[ii] = hit * 1.0 / (pp + gp - hit)
    return jac


def dice(conf_matrix):

    num_cls = conf_matrix.shape[0]
    dic = np.zeros(num_cls)
    for ii in range(num_cls):
        pp = np.sum(conf_matrix[:,ii])
        gp = np.sum(conf_matrix[ii,:])
        hit = conf_matrix[ii,ii]
        if (pp + gp) == 0:
            dic[ii] = 0
        else:
            dic[ii] = 2.0 * hit / (pp + gp)
    return dic


def dice_eval(compact_pred, labels, n_class):

    dice_arr = []
    dice = 0
    eps = 1e-7
    pred = tf.one_hot(compact_pred, depth = n_class, axis = -1)
    for i in xrange(n_class):
        inse = tf.reduce_sum(pred[:, :, :, i] * labels[:, :, :, i])
        union = tf.reduce_sum(pred[:, :, :, i]) + tf.reduce_sum(labels[:, :, :, i])
        dice = dice + 2.0 * inse / (union + eps)
        dice_arr.append(2.0 * inse / (union + eps))

    return dice_arr