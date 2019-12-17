

import tensorflow as tf
import layers
import json

with open('./config_param.json') as config_file:
    config = json.load(config_file)

BATCH_SIZE = int(config['batch_size'])
POOL_SIZE = int(config['pool_size'])

# The height of each image.
IMG_HEIGHT = 256

# The width of each image.
IMG_WIDTH = 256

ngf = 32
ndf = 64


def get_outputs(inputs, skip=False, is_training=True, keep_rate=0.75):
    images_a = inputs['images_a']
    images_b = inputs['images_b']
    fake_pool_a = inputs['fake_pool_a']
    fake_pool_b = inputs['fake_pool_b']

    with tf.variable_scope("Model", reuse=tf.AUTO_REUSE) as scope:

        current_discriminator = discriminator
        current_encoder = build_encoder
        current_decoder = build_decoder
        current_segmenter = build_segmenter

        prob_real_a_is_real, prob_real_a_aux = discriminator_aux(images_a, "d_A")
        prob_real_b_is_real = current_discriminator(images_b, "d_B")

        fake_images_b = build_generator_resnet_9blocks(images_a, images_a, name='g_A', skip=skip)
        latent_b, latent_b_ll = current_encoder(images_b, name='e_B', skip=skip, is_training=is_training, keep_rate=keep_rate)

        fake_images_a = current_decoder(latent_b, images_b, name='de_B', skip=skip)

        pred_mask_b = current_segmenter(latent_b, name='s_B', keep_rate=keep_rate)
        pred_mask_b_ll = current_segmenter(latent_b_ll, name='s_B_ll', keep_rate=keep_rate)


        prob_fake_a_is_real, prob_fake_a_aux_is_real = discriminator_aux(fake_images_a, "d_A")
        prob_fake_b_is_real = current_discriminator(fake_images_b, "d_B")

        latent_fake_b, latent_fake_b_ll = current_encoder(fake_images_b, 'e_B', skip=skip, is_training=is_training, keep_rate=keep_rate)
        cycle_images_b = build_generator_resnet_9blocks(fake_images_a, fake_images_a, 'g_A', skip=skip)

        cycle_images_a = current_decoder(latent_fake_b, fake_images_b, 'de_B', skip=skip)

        pred_mask_fake_b = current_segmenter(latent_fake_b, 's_B', keep_rate=keep_rate)
        pred_mask_fake_b_ll = current_segmenter(latent_fake_b_ll, 's_B_ll', keep_rate=keep_rate)

        prob_fake_pool_a_is_real, prob_fake_pool_a_aux_is_real = discriminator_aux(fake_pool_a, "d_A")
        prob_fake_pool_b_is_real = current_discriminator(fake_pool_b, "d_B")

        prob_cycle_a_is_real, prob_cycle_a_aux_is_real = discriminator_aux(cycle_images_a, "d_A")

        prob_pred_mask_fake_b_is_real = current_discriminator(pred_mask_fake_b, name="d_P")
        prob_pred_mask_b_is_real = current_discriminator(pred_mask_b, 'd_P')

        prob_pred_mask_fake_b_ll_is_real = current_discriminator(pred_mask_fake_b_ll, name="d_P_ll")
        prob_pred_mask_b_ll_is_real = current_discriminator(pred_mask_b_ll, 'd_P_ll')

    return {
        'prob_real_a_is_real': prob_real_a_is_real,
        'prob_real_b_is_real': prob_real_b_is_real,
        'prob_fake_a_is_real': prob_fake_a_is_real,
        'prob_fake_b_is_real': prob_fake_b_is_real,
        'prob_fake_pool_a_is_real': prob_fake_pool_a_is_real,
        'prob_fake_pool_b_is_real': prob_fake_pool_b_is_real,
        'cycle_images_a': cycle_images_a,
        'cycle_images_b': cycle_images_b,
        'fake_images_a': fake_images_a,
        'fake_images_b': fake_images_b,
        'pred_mask_a': pred_mask_b,
        'pred_mask_b': pred_mask_b,
        'pred_mask_b_ll': pred_mask_b_ll,
        'pred_mask_fake_a': pred_mask_fake_b,
        'pred_mask_fake_b': pred_mask_fake_b,
        'pred_mask_fake_b_ll': pred_mask_fake_b_ll,
        'prob_pred_mask_fake_b_is_real': prob_pred_mask_fake_b_is_real,
        'prob_pred_mask_b_is_real': prob_pred_mask_b_is_real,
        'prob_pred_mask_fake_b_ll_is_real': prob_pred_mask_fake_b_ll_is_real,
        'prob_pred_mask_b_ll_is_real': prob_pred_mask_b_ll_is_real,
        'prob_fake_a_aux_is_real': prob_fake_a_aux_is_real,
        'prob_fake_pool_a_aux_is_real': prob_fake_pool_a_aux_is_real,
        'prob_cycle_a_aux_is_real': prob_cycle_a_aux_is_real,
    }


def build_resnet_block(inputres, dim, name="resnet", padding="REFLECT", norm_type=None, is_training=True, keep_rate=0.75):
    with tf.variable_scope(name):
        out_res = tf.pad(inputres, [[0, 0], [1, 1], [1, 1], [0, 0]], padding)
        out_res = layers.general_conv2d(out_res, dim, 3, 3, 1, 1, 0.01, "VALID", "c1", norm_type=norm_type, is_training=is_training, keep_rate=keep_rate)
        out_res = tf.pad(out_res, [[0, 0], [1, 1], [1, 1], [0, 0]], padding)
        out_res = layers.general_conv2d(out_res, dim, 3, 3, 1, 1, 0.01, "VALID", "c2", do_relu=False, norm_type=norm_type, is_training=is_training, keep_rate=keep_rate)

        return tf.nn.relu(out_res + inputres)


def build_resnet_block_ins(inputres, dim, name="resnet", padding="REFLECT"):
    with tf.variable_scope(name):
        out_res = tf.pad(inputres, [[0, 0], [1, 1], [1, 1], [0, 0]], padding)
        out_res = layers.general_conv2d_ga(out_res, dim, 3, 3, 1, 1, 0.02, "VALID", "c1", norm_type='Ins')
        out_res = tf.pad(out_res, [[0, 0], [1, 1], [1, 1], [0, 0]], padding)
        out_res = layers.general_conv2d_ga(out_res, dim, 3, 3, 1, 1, 0.02, "VALID", "c2", do_relu=False, norm_type='Ins')

        return tf.nn.relu(out_res + inputres)


def build_resnet_block_ds(inputres, dim_in, dim_out, name="resnet", padding="REFLECT", norm_type=None, is_training=True, keep_rate=0.75):

    with tf.variable_scope(name):
        out_res = tf.pad(inputres, [[0, 0], [1, 1], [1, 1], [0, 0]], padding)
        out_res = layers.general_conv2d(out_res, dim_out, 3, 3, 1, 1, 0.01, "VALID", "c1", norm_type=norm_type, is_training=is_training, keep_rate=keep_rate)
        out_res = tf.pad(out_res, [[0, 0], [1, 1], [1, 1], [0, 0]], padding)
        out_res = layers.general_conv2d(out_res, dim_out, 3, 3, 1, 1, 0.01, "VALID", "c2", do_relu=False, norm_type=norm_type, is_training=is_training, keep_rate=keep_rate)

        inputres = tf.pad(inputres, [[0, 0], [0, 0], [0, 0], [(dim_out - dim_in) // 2, (dim_out - dim_in) // 2]], padding)

        return tf.nn.relu(out_res + inputres)


def build_drn_block(inputdrn, dim, name="drn", padding="REFLECT", norm_type=None, is_training=True, keep_rate=0.75):

    with tf.variable_scope(name):
        out_drn = tf.pad(inputdrn, [[0, 0], [2, 2], [2, 2], [0, 0]], padding)
        out_drn = layers.dilate_conv2d(out_drn, dim, dim, 3, 3, 2, 0.01, "VALID", "c1", norm_type=norm_type, is_training=is_training, keep_rate=keep_rate)
        out_drn = tf.pad(out_drn, [[0, 0], [2, 2], [2, 2], [0, 0]], padding)
        out_drn = layers.dilate_conv2d(out_drn, dim, dim, 3, 3, 2, 0.01, "VALID", "c2", do_relu=False, norm_type=norm_type, is_training=is_training, keep_rate=keep_rate)

        return tf.nn.relu(out_drn + inputdrn)


def build_drn_block_ds(inputdrn, dim_in, dim_out, name='drn_ds', padding="REFLECT", norm_type=None, is_training=True, keep_rate=0.75):
    with tf.variable_scope(name):
        out_drn = tf.pad(inputdrn, [[0,0], [2,2], [2,2], [0,0]], padding)
        out_drn = layers.dilate_conv2d(out_drn, dim_in, dim_out, 3, 3, 2, 0.01, 'VALID', "c1", norm_type=norm_type, is_training=is_training, keep_rate=keep_rate)
        out_drn = tf.pad(out_drn, [[0,0], [2,2], [2,2], [0,0]], padding)
        out_drn = layers.dilate_conv2d(out_drn, dim_out, dim_out, 3, 3, 2, 0.01, 'VALID', "c2", do_relu=False, norm_type=norm_type, is_training=is_training, keep_rate=keep_rate)

        inputdrn = tf.pad(inputdrn, [[0,0], [0,0], [0, 0], [(dim_out-dim_in)//2,(dim_out-dim_in)//2]], padding)

        return tf.nn.relu(out_drn + inputdrn)


def build_generator_resnet_9blocks(inputgen, inputimg, name="generator", skip=False):
    with tf.variable_scope(name):
        f = 7
        ks = 3
        padding = "CONSTANT"

        pad_input = tf.pad(inputgen, [[0, 0], [ks, ks], [ks, ks], [0, 0]], padding)
        o_c1 = layers.general_conv2d_ga(pad_input, ngf, f, f, 1, 1, 0.02, name="c1", norm_type='Ins')
        o_c2 = layers.general_conv2d_ga(o_c1, ngf * 2, ks, ks, 2, 2, 0.02, "SAME", "c2", norm_type='Ins')
        o_c3 = layers.general_conv2d_ga(o_c2, ngf * 4, ks, ks, 2, 2, 0.02, "SAME", "c3", norm_type='Ins')

        o_r1 = build_resnet_block_ins(o_c3, ngf * 4, "r1", padding)
        o_r2 = build_resnet_block_ins(o_r1, ngf * 4, "r2", padding)
        o_r3 = build_resnet_block_ins(o_r2, ngf * 4, "r3", padding)
        o_r4 = build_resnet_block_ins(o_r3, ngf * 4, "r4", padding)
        o_r5 = build_resnet_block_ins(o_r4, ngf * 4, "r5", padding)
        o_r6 = build_resnet_block_ins(o_r5, ngf * 4, "r6", padding)
        o_r7 = build_resnet_block_ins(o_r6, ngf * 4, "r7", padding)
        o_r8 = build_resnet_block_ins(o_r7, ngf * 4, "r8", padding)
        o_r9 = build_resnet_block_ins(o_r8, ngf * 4, "r9", padding)

        o_c4 = layers.general_deconv2d(o_r9, [BATCH_SIZE, 128, 128, ngf * 2], ngf * 2, ks, ks, 2, 2, 0.02, "SAME", "c4", norm_type='Ins')
        o_c5 = layers.general_deconv2d(o_c4, [BATCH_SIZE, 256, 256, ngf], ngf, ks, ks, 2, 2, 0.02, "SAME", "c5", norm_type='Ins')
        o_c6 = layers.general_conv2d_ga(o_c5, 1, f, f, 1, 1, 0.02, "SAME", "c6", do_norm=False, do_relu=False)

        if skip is True:
            out_gen = tf.nn.tanh(inputimg + o_c6, "t1")
        else:
            out_gen = tf.nn.tanh(o_c6, "t1")

        return out_gen


def build_encoder(inputen, name='encoder', skip=False, is_training=True, keep_rate=0.75):
    with tf.variable_scope(name):
        fb = 16
        k1 = 3
        padding = "CONSTANT"

        o_c1 = layers.general_conv2d(inputen, fb, 7, 7, 1, 1, 0.01, 'SAME', name="c1", norm_type="Batch", is_training=is_training, keep_rate=keep_rate)
        o_r1 = build_resnet_block(o_c1, fb, "r1", padding, norm_type='Batch', is_training=is_training, keep_rate=keep_rate)
        out1 = tf.nn.max_pool(o_r1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        o_r2 = build_resnet_block_ds(out1, fb, fb*2, "r2", padding, norm_type='Batch', is_training=is_training, keep_rate=keep_rate)
        out2 = tf.nn.max_pool(o_r2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        o_r3 = build_resnet_block_ds(out2, fb*2, fb*4, 'r3', padding, norm_type='Batch', is_training=is_training, keep_rate=keep_rate)
        o_r4 = build_resnet_block(o_r3, fb*4, 'r4', padding, norm_type='Batch', is_training=is_training, keep_rate=keep_rate)
        out3 = tf.nn.max_pool(o_r4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        o_r5 = build_resnet_block_ds(out3, fb*4, fb*8, 'r5', padding, norm_type='Batch', is_training=is_training, keep_rate=keep_rate)
        o_r6 = build_resnet_block(o_r5, fb*8, 'r6', padding, norm_type='Batch', is_training=is_training, keep_rate=keep_rate)

        o_r7 = build_resnet_block_ds(o_r6, fb*8, fb*16, 'r7', padding, norm_type='Batch', is_training=is_training, keep_rate=keep_rate)
        o_r8 = build_resnet_block(o_r7, fb*16, 'r8', padding, norm_type='Batch', is_training=is_training, keep_rate=keep_rate)

        o_r9 = build_resnet_block(o_r8, fb*16, 'r9', padding, norm_type='Batch', is_training=is_training, keep_rate=keep_rate)
        o_r10 = build_resnet_block(o_r9, fb * 16, 'r10', padding, norm_type='Batch', is_training=is_training, keep_rate=keep_rate)

        o_r11 = build_resnet_block_ds(o_r10, fb * 16, fb * 32, 'r11', padding, norm_type='Batch', is_training=is_training, keep_rate=keep_rate)
        o_r12 = build_resnet_block(o_r11, fb * 32, 'r12', padding, norm_type='Batch', is_training=is_training, keep_rate=keep_rate)

        o_d1 = build_drn_block(o_r12, fb*32, 'd1', padding, norm_type='Batch', is_training=is_training, keep_rate=keep_rate)
        o_d2 = build_drn_block(o_d1, fb*32, 'd2', padding, norm_type='Batch', is_training=is_training, keep_rate=keep_rate)

        o_c2 = layers.general_conv2d(o_d2, fb*32, k1, k1, 1, 1, 0.01, 'SAME', 'c2', norm_type='Batch', is_training=is_training,keep_rate=keep_rate)
        o_c3 = layers.general_conv2d(o_c2, fb*32, k1, k1, 1, 1, 0.01, 'SAME', 'c3', norm_type='Batch', is_training=is_training, keep_rate=keep_rate)


        return o_c3, o_r12


def build_decoder(inputde, inputimg, name='decoder', skip=False):
    with tf.variable_scope(name):
        f = 7
        ks = 3
        padding = "CONSTANT"

        o_c1 = layers.general_conv2d(inputde, ngf * 4, ks, ks, 1, 1, 0.02, "SAME", "c1", norm_type='Ins')
        o_r1 = build_resnet_block(o_c1, ngf * 4, "r1", padding, norm_type='Ins')
        o_r2 = build_resnet_block(o_r1, ngf * 4, "r2", padding, norm_type='Ins')
        o_r3 = build_resnet_block(o_r2, ngf * 4, "r3", padding, norm_type='Ins')
        o_r4 = build_resnet_block(o_r3, ngf * 4, "r4", padding, norm_type='Ins')
        o_c3 = layers.general_deconv2d(o_r4, [BATCH_SIZE, 64, 64, ngf * 2], ngf * 2, ks, ks, 2, 2, 0.02, "SAME", "c3", norm_type='Ins')
        o_c4 = layers.general_deconv2d(o_c3, [BATCH_SIZE, 128, 128, ngf * 2], ngf * 2, ks, ks, 2, 2, 0.02, "SAME", "c4", norm_type='Ins')
        o_c5 = layers.general_deconv2d(o_c4, [BATCH_SIZE, 256, 256, ngf], ngf, ks, ks, 2, 2, 0.02, "SAME", "c5", norm_type='Ins')
        o_c6 = layers.general_conv2d(o_c5, 1, f, f, 1, 1, 0.02, "SAME", "c6", do_norm=False, do_relu=False)

        if skip is True:
            out_gen = tf.nn.tanh(inputimg + o_c6, "t1")
        else:
            out_gen = tf.nn.tanh(o_c6, "t1")

        return out_gen


def build_segmenter(inputse, name='segmenter', keep_rate=0.75):
    with tf.variable_scope(name):

        k1 = 1

        o_c8 = layers.general_conv2d(inputse, 5, k1, k1, 1, 1, 0.01, 'SAME', 'c8', do_norm=False, do_relu=False, keep_rate=keep_rate)
        out_seg = tf.image.resize_images(o_c8, (256, 256))

        return out_seg


def discriminator(inputdisc, name="discriminator"):
    with tf.variable_scope(name):
        f = 4
        padw = 2

        pad_input = tf.pad(inputdisc, [[0, 0], [padw, padw], [padw, padw], [0, 0]], "CONSTANT")
        o_c1 = layers.general_conv2d(pad_input, ndf, f, f, 2, 2, 0.02, "VALID", "c1", do_norm=False, relufactor=0.2, norm_type='Ins')

        pad_o_c1 = tf.pad(o_c1, [[0, 0], [padw, padw], [padw, padw], [0, 0]], "CONSTANT")
        o_c2 = layers.general_conv2d(pad_o_c1, ndf * 2, f, f, 2, 2, 0.02, "VALID", "c2", relufactor=0.2, norm_type='Ins')

        pad_o_c2 = tf.pad(o_c2, [[0, 0], [padw, padw], [padw, padw], [0, 0]], "CONSTANT")
        o_c3 = layers.general_conv2d(pad_o_c2, ndf * 4, f, f, 2, 2, 0.02, "VALID", "c3", relufactor=0.2, norm_type='Ins')

        pad_o_c3 = tf.pad(o_c3, [[0, 0], [padw, padw], [padw, padw], [0, 0]], "CONSTANT")
        o_c4 = layers.general_conv2d(pad_o_c3, ndf * 8, f, f, 1, 1, 0.02, "VALID", "c4", relufactor=0.2, norm_type='Ins')

        pad_o_c4 = tf.pad(o_c4, [[0, 0], [padw, padw], [padw, padw], [0, 0]], "CONSTANT")
        o_c5 = layers.general_conv2d(pad_o_c4, 1, f, f, 1, 1, 0.02, "VALID", "c5", do_norm=False, do_relu=False)

        return o_c5


def discriminator_aux(inputdisc, name="discriminator"):
    with tf.variable_scope(name):
        f = 4
        padw = 2

        pad_input = tf.pad(inputdisc, [[0, 0], [padw, padw], [padw, padw], [0, 0]], "CONSTANT")
        o_c1 = layers.general_conv2d(pad_input, ndf, f, f, 2, 2, 0.02, "VALID", "c1", do_norm=False, relufactor=0.2, norm_type='Ins')

        pad_o_c1 = tf.pad(o_c1, [[0, 0], [padw, padw], [padw, padw], [0, 0]], "CONSTANT")
        o_c2 = layers.general_conv2d(pad_o_c1, ndf * 2, f, f, 2, 2, 0.02, "VALID", "c2", relufactor=0.2, norm_type='Ins')

        pad_o_c2 = tf.pad(o_c2, [[0, 0], [padw, padw], [padw, padw], [0, 0]], "CONSTANT")
        o_c3 = layers.general_conv2d(pad_o_c2, ndf * 4, f, f, 2, 2, 0.02, "VALID", "c3", relufactor=0.2, norm_type='Ins')

        pad_o_c3 = tf.pad(o_c3, [[0, 0], [padw, padw], [padw, padw], [0, 0]], "CONSTANT")
        o_c4 = layers.general_conv2d(pad_o_c3, ndf * 8, f, f, 1, 1, 0.02, "VALID", "c4", relufactor=0.2, norm_type='Ins')

        pad_o_c4 = tf.pad(o_c4, [[0, 0], [padw, padw], [padw, padw], [0, 0]], "CONSTANT")
        o_c5 = layers.general_conv2d(pad_o_c4, 2, f, f, 1, 1, 0.02, "VALID", "c5", do_norm=False, do_relu=False)

        return tf.expand_dims(o_c5[...,0], axis=3), tf.expand_dims(o_c5[...,1], axis=3)