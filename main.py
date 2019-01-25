"""Code for training SIFA."""
from datetime import datetime
import json
import numpy as np
import os
import random

import tensorflow as tf

import data_loader, losses, model
import cv2

import csv
from stats_func import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


source_train_pth = './data/datalist/training_A.txt'
target_train_pth = './data/datalist/training_B.txt'
source_val_pth = './data/datalist/validation_A.txt'
target_val_pth = './data/datalist/validation_B.txt'

evaluation_interval = 10
save_interval = 300
num_cls = 5
keep_rate_value=0.75
is_training_value=True

BATCH_SIZE = 8

class SIFA:
    """The SIFA module."""

    def __init__(self, pool_size, lambda_a,
                 lambda_b, output_root_dir, to_restore,
                 base_lr, max_step,
                 checkpoint_dir, do_flipping, skip):
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

        self._source_train_pth = source_train_pth
        self._target_train_pth = target_train_pth
        self._source_val_pth = source_val_pth
        self._target_val_pth = target_val_pth
        self._num_cls = num_cls
        self.keep_rate=tf.placeholder(tf.float32, shape=())
        self.is_training = tf.placeholder(tf.bool, shape=())
        self._pool_size = pool_size
        self._size_before_crop = 256
        self._lambda_a = lambda_a
        self._lambda_b = lambda_b
        self._output_dir = os.path.join(output_root_dir, current_time)
        self._images_dir = os.path.join(self._output_dir, 'imgs')
        self._num_imgs_to_save = 20
        self._to_restore = to_restore
        self._base_lr = base_lr
        self._max_step = max_step
        self._checkpoint_dir = checkpoint_dir
        self._do_flipping = do_flipping
        self._skip = skip

        self.fake_images_A = np.zeros(
            (self._pool_size, BATCH_SIZE, model.IMG_HEIGHT, model.IMG_WIDTH, 1)
        )
        self.fake_images_B = np.zeros(
            (self._pool_size, BATCH_SIZE, model.IMG_HEIGHT, model.IMG_WIDTH, 1)
        )

    def model_setup(self):

        self.input_a = tf.placeholder(
            tf.float32, [
                None,
                model.IMG_WIDTH,
                model.IMG_HEIGHT,
                3
            ], name="input_A")
        self.input_b = tf.placeholder(
            tf.float32, [
                None,
                model.IMG_WIDTH,
                model.IMG_HEIGHT,
                3
            ], name="input_B")
        self.fake_pool_A = tf.placeholder(
            tf.float32, [
                None,
                model.IMG_WIDTH,
                model.IMG_HEIGHT,
                1
            ], name="fake_pool_A")
        self.fake_pool_B = tf.placeholder(
            tf.float32, [
                None,
                model.IMG_WIDTH,
                model.IMG_HEIGHT,
                1
            ], name="fake_pool_B")
        self.gt_a = tf.placeholder(
            tf.float32, [
                None,
                model.IMG_WIDTH,
                model.IMG_HEIGHT,
                5
            ], name="gt_A")
        self.gt_b = tf.placeholder(
            tf.float32, [
                None,
                model.IMG_WIDTH,
                model.IMG_HEIGHT,
                5
            ], name="gt_B")

        self.global_step = tf.train.get_or_create_global_step()

        self.num_fake_inputs = 0

        self.learning_rate = tf.placeholder(tf.float32, shape=[], name="lr")
        self.learning_rate_seg = tf.placeholder(tf.float32, shape=[], name="lr_seg")

        self.lr_gan_summ = tf.summary.scalar("lr_gan", self.learning_rate)
        self.lr_seg_summ = tf.summary.scalar("lr_seg", self.learning_rate_seg)

        inputs = {
            'images_a': self.input_a,
            'images_b': self.input_b,
            'fake_pool_a': self.fake_pool_A,
            'fake_pool_b': self.fake_pool_B,
            'gt_a': self.gt_a,
        }

        outputs = model.get_outputs(
            inputs, skip=self._skip, is_training=self.is_training, keep_rate=self.keep_rate)

        self.prob_real_a_is_real = outputs['prob_real_a_is_real']
        self.prob_real_b_is_real = outputs['prob_real_b_is_real']
        self.fake_images_a = outputs['fake_images_a']
        self.fake_images_b = outputs['fake_images_b']
        self.prob_fake_a_is_real = outputs['prob_fake_a_is_real']
        self.prob_fake_b_is_real = outputs['prob_fake_b_is_real']

        self.cycle_images_a = outputs['cycle_images_a']
        self.cycle_images_b = outputs['cycle_images_b']

        self.prob_fake_pool_a_is_real = outputs['prob_fake_pool_a_is_real']
        self.prob_fake_pool_b_is_real = outputs['prob_fake_pool_b_is_real']
        self.pred_mask_a = outputs['pred_mask_a']
        self.pred_mask_b = outputs['pred_mask_b']
        self.pred_mask_fake_a = outputs['pred_mask_fake_a']
        self.pred_mask_fake_b = outputs['pred_mask_fake_b']
        self.prob_fea_fake_b_is_real = outputs['prob_fea_fake_b_is_real']
        self.prob_fea_b_is_real = outputs['prob_fea_b_is_real']

        self.prob_real_a_aux = outputs['prob_real_a_aux']
        self.prob_fake_a_aux_is_real = outputs['prob_fake_a_aux_is_real']
        self.prob_fake_pool_a_aux_is_real = outputs['prob_fake_pool_a_aux_is_real']
        self.prob_cycle_a_is_real = outputs['prob_cycle_a_is_real']
        self.prob_cycle_a_aux_is_real = outputs['prob_cycle_a_aux_is_real']

        self.predicter_fake_b = pixel_wise_softmax_2(self.pred_mask_fake_b)
        self.compact_pred_fake_b = tf.argmax(self.predicter_fake_b, 3)
        self.compact_y_fake_b = tf.argmax(self.gt_a, 3)
        self.confusion_matrix_fake_b = tf.confusion_matrix(tf.reshape(self.compact_y_fake_b, [-1]),
                                                           tf.reshape(self.compact_pred_fake_b, [-1]),
                                                           num_classes=self._num_cls)

        self.predicter_b = pixel_wise_softmax_2(self.pred_mask_b)
        self.compact_pred_b = tf.argmax(self.predicter_b, 3)
        self.compact_y_b = tf.argmax(self.gt_b, 3)
        self.confusion_matrix_b = tf.confusion_matrix(tf.reshape(self.compact_y_b, [-1]),
                                                      tf.reshape(self.compact_pred_b, [-1]), num_classes=self._num_cls)

    def compute_losses(self):

        cycle_consistency_loss_a = \
            self._lambda_a * losses.cycle_consistency_loss(
                real_images=tf.expand_dims(self.input_a[:,:,:,1], axis=3), generated_images=self.cycle_images_a,
            )
        cycle_consistency_loss_b = \
            self._lambda_b * losses.cycle_consistency_loss(
                real_images=tf.expand_dims(self.input_b[:,:,:,1], axis=3), generated_images=self.cycle_images_b,
            )

        lsgan_loss_a = losses.lsgan_loss_generator(self.prob_fake_a_is_real)
        lsgan_loss_b = losses.lsgan_loss_generator(self.prob_fake_b_is_real)
        lsgan_loss_f = losses.lsgan_loss_generator(self.prob_fea_b_is_real)
        lsgan_loss_a_aux = losses.lsgan_loss_generator(self.prob_fake_a_aux_is_real)

        ce_loss_b, dice_loss_b = losses.task_loss(self.pred_mask_fake_b, self.gt_a)

        l2_loss_b = tf.add_n([0.0001 * tf.nn.l2_loss(v) for v in tf.trainable_variables() if '/s_B/' in v.name or '/e_B/' in v.name])


        g_loss_A = cycle_consistency_loss_a + cycle_consistency_loss_b + lsgan_loss_b
        g_loss_B = cycle_consistency_loss_b + cycle_consistency_loss_a + lsgan_loss_a

        self.loss_f_weight = tf.placeholder(tf.float32, shape=[], name="loss_f_weight")
        self.loss_f_weight_summ = tf.summary.scalar("loss_f_weight", self.loss_f_weight)
        seg_loss_B = ce_loss_b + dice_loss_b + l2_loss_b + 0.1*g_loss_B + self.loss_f_weight*lsgan_loss_f + 0.1*lsgan_loss_a_aux


        d_loss_A = losses.lsgan_loss_discriminator(
            prob_real_is_real=self.prob_real_a_is_real,
            prob_fake_is_real=self.prob_fake_pool_a_is_real,
        )
        d_loss_A_aux = losses.lsgan_loss_discriminator(
            prob_real_is_real=self.prob_cycle_a_aux_is_real,
            prob_fake_is_real=self.prob_fake_pool_a_aux_is_real,
        )
        d_loss_A = d_loss_A + d_loss_A_aux
        d_loss_B = losses.lsgan_loss_discriminator(
            prob_real_is_real=self.prob_real_b_is_real,
            prob_fake_is_real=self.prob_fake_pool_b_is_real,
        )
        d_loss_F = losses.lsgan_loss_discriminator(
            prob_real_is_real=self.prob_fea_fake_b_is_real,
            prob_fake_is_real=self.prob_fea_b_is_real,
        )

        optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5)
        optimizer_seg = tf.train.AdamOptimizer(self.learning_rate_seg)

        self.model_vars = tf.trainable_variables()

        d_A_vars = [var for var in self.model_vars if '/d_A/' in var.name]
        d_B_vars = [var for var in self.model_vars if '/d_B/' in var.name]
        g_A_vars = [var for var in self.model_vars if '/g_A/' in var.name]
        e_B_vars = [var for var in self.model_vars if '/e_B/' in var.name]
        de_B_vars = [var for var in self.model_vars if '/de_B/' in var.name]
        s_B_vars = [var for var in self.model_vars if '/s_B/' in var.name]
        d_F_vars = [var for var in self.model_vars if '/d_F/' in var.name]

        self.d_A_trainer = optimizer.minimize(d_loss_A, var_list=d_A_vars)
        self.d_B_trainer = optimizer.minimize(d_loss_B, var_list=d_B_vars)
        self.g_A_trainer = optimizer.minimize(g_loss_A, var_list=g_A_vars)
        self.g_B_trainer = optimizer.minimize(g_loss_B, var_list=de_B_vars)
        self.s_B_trainer = optimizer_seg.minimize(seg_loss_B, var_list=e_B_vars+s_B_vars)
        self.d_F_trainer = optimizer.minimize(d_loss_F, var_list=d_F_vars)

        for var in self.model_vars:
            print(var.name)

        # Summary variables for tensorboard
        self.g_A_loss_summ = tf.summary.scalar("g_A_loss", g_loss_A)
        self.g_B_loss_summ = tf.summary.scalar("g_B_loss", g_loss_B)
        self.d_A_loss_summ = tf.summary.scalar("d_A_loss", d_loss_A)
        self.d_B_loss_summ = tf.summary.scalar("d_B_loss", d_loss_B)
        self.ce_B_loss_summ = tf.summary.scalar("ce_B_loss", ce_loss_b)
        self.dice_B_loss_summ = tf.summary.scalar("dice_B_loss", dice_loss_b)
        self.l2_B_loss_summ = tf.summary.scalar("l2_B_loss", l2_loss_b)
        self.s_B_loss_summ = tf.summary.scalar("s_B_loss", seg_loss_B)
        self.s_B_loss_merge_summ = tf.summary.merge([self.ce_B_loss_summ, self.dice_B_loss_summ, self.l2_B_loss_summ, self.s_B_loss_summ])
        self.d_F_loss_summ = tf.summary.scalar("d_F_loss", d_loss_F)

    def save_images(self, sess, epoch):

        if not os.path.exists(self._images_dir):
            os.makedirs(self._images_dir)

        names = ['inputA_', 'inputB_', 'fakeA_',
                 'fakeB_', 'cycA_', 'cycB_']

        with open(os.path.join(self._output_dir, 'epoch_' + str(epoch) + '.html'), 'w') as v_html:
            for i in range(0, self._num_imgs_to_save):
                # print("Saving image {}/{}".format(i, self._num_imgs_to_save))
                images_i, images_j, gts_i, gts_j = sess.run(self.inputs)
                inputs = {
                    'images_i': images_i,
                    'images_j': images_j,
                    'gts_i': gts_i,
                    'gts_j': gts_j,
                }


                fake_A_temp, fake_B_temp, cyc_A_temp, cyc_B_temp = sess.run([
                    self.fake_images_a,
                    self.fake_images_b,
                    self.cycle_images_a,
                    self.cycle_images_b
                ], feed_dict={
                    self.input_a: inputs['images_i'],
                    self.input_b: inputs['images_j'],
                    self.is_training:is_training_value,
                    self.keep_rate: keep_rate_value,
                })


                tensors = [inputs['images_i'], inputs['images_j'],
                           fake_B_temp, fake_A_temp, cyc_A_temp, cyc_B_temp]

                for name, tensor in zip(names, tensors):
                    image_name = name + str(epoch) + "_" + str(i) + ".jpg"
                    cv2.imwrite(os.path.join(self._images_dir, image_name), ((tensor[0] + 1) * 127.5).astype(np.uint8).squeeze())
                    v_html.write("<img src=\"" + os.path.join('imgs', image_name) + "\">")
                v_html.write("<br>")


    def fake_image_pool(self, num_fakes, fake, fake_pool):

        if num_fakes < self._pool_size:
            fake_pool[num_fakes] = fake
            return fake
        else:
            p = random.random()
            if p > 0.5:
                random_id = random.randint(0, self._pool_size - 1)
                temp = fake_pool[random_id]
                fake_pool[random_id] = fake
                return temp
            else:
                return fake

    def train(self):
        """Training Function."""
        # Load Dataset from the dataset folder
        self.inputs = data_loader.load_data(self._source_train_pth, self._target_train_pth, True)
        self.inputs_val = data_loader.load_data(self._source_val_pth, self._target_val_pth, True)

        # Build the network
        self.model_setup()

        # Loss function calculations
        self.compute_losses()

        # Initializing the global variables
        init = (tf.global_variables_initializer(),
                tf.local_variables_initializer())
        saver = tf.train.Saver(max_to_keep=40)

        with open(self._source_train_pth, 'r') as fp:
            rows_s = fp.readlines()
        with open(self._target_train_pth, 'r') as fp:
            rows_t = fp.readlines()

        max_images = max(len(rows_s), len(rows_t))

        gpu_options = tf.GPUOptions(allow_growth=True)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            sess.run(init)


            # Restore the model to run the model from last checkpoint
            if self._to_restore:
                chkpt_fname = tf.train.latest_checkpoint(self._checkpoint_dir)
                saver.restore(sess, chkpt_fname)

            writer = tf.summary.FileWriter(self._output_dir)
            writer_val = tf.summary.FileWriter(self._output_dir+'/val')

            if not os.path.exists(self._output_dir):
                os.makedirs(self._output_dir)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            # Training Loop
            curr_lr_seg = 0.001
            cnt = -1

            for epoch in range(sess.run(self.global_step), self._max_step):
                print("In the epoch ", epoch)

                curr_lr = self._base_lr

                if epoch < 5:
                    loss_f_weight_value = 0.0
                elif epoch < 7:
                    loss_f_weight_value = 0.1 * (epoch - 4.0) / (7.0 - 4.0)
                else:
                    loss_f_weight_value = 0.1

                self.save_images(sess, epoch)

                if epoch > 0 and epoch%2==0:
                    curr_lr_seg = np.multiply(curr_lr_seg, 0.9)

                max_inter = np.uint16(np.floor(max_images/BATCH_SIZE))

                for i in range(0, max_inter):
                    cnt += 1

                    print("Processing batch {}/{}".format(i, max_inter))

                    images_i, images_j, gts_i, gts_j = sess.run(self.inputs)
                    inputs = {
                        'images_i': images_i,
                        'images_j': images_j,
                        'gts_i': gts_i,
                        'gts_j': gts_j,
                    }
                    images_i_val, images_j_val, gts_i_val, gts_j_val = sess.run(self.inputs_val)
                    inputs_val = {
                        'images_i_val': images_i_val,
                        'images_j_val': images_j_val,
                        'gts_i_val': gts_i_val,
                        'gts_j_val': gts_j_val,
                    }

                    # Optimizing the G_A network
                    _, fake_B_temp, summary_str = sess.run(
                        [self.g_A_trainer,
                         self.fake_images_b,
                         self.g_A_loss_summ],
                        feed_dict={
                            self.input_a:
                                inputs['images_i'],
                            self.input_b:
                                inputs['images_j'],
                            self.gt_a:
                                inputs['gts_i'],
                            self.learning_rate: curr_lr,
                            self.keep_rate:keep_rate_value,
                            self.is_training:is_training_value,
                            self.loss_f_weight: loss_f_weight_value,
                        }
                    )
                    writer.add_summary(summary_str, epoch * max_inter + i)

                    fake_B_temp1 = self.fake_image_pool(
                        self.num_fake_inputs, fake_B_temp, self.fake_images_B)

                    # Optimizing the D_B network
                    _, summary_str = sess.run(
                        [self.d_B_trainer, self.d_B_loss_summ],
                        feed_dict={
                            self.input_a:
                                inputs['images_i'],
                            self.input_b:
                                inputs['images_j'],
                            self.learning_rate: curr_lr,
                            self.fake_pool_B: fake_B_temp1,
                            self.keep_rate: keep_rate_value,
                            self.is_training: is_training_value,
                            self.loss_f_weight: loss_f_weight_value,
                        }
                    )
                    writer.add_summary(summary_str, epoch * max_inter + i)

                    # Optimizing the S_B network
                    _, summary_str = sess.run(
                        [self.s_B_trainer, self.s_B_loss_merge_summ],
                        feed_dict={
                            self.input_a:
                                inputs['images_i'],
                            self.input_b:
                                inputs['images_j'],
                            self.gt_a:
                                inputs['gts_i'],
                            self.learning_rate_seg: curr_lr_seg,
                            self.keep_rate: keep_rate_value,
                            self.is_training: is_training_value,
                            self.loss_f_weight: loss_f_weight_value,
                        }

                    )
                    writer.add_summary(summary_str, epoch * max_inter + i)

                    # Optimizing the G_B network
                    _, fake_A_temp, summary_str = sess.run(
                        [self.g_B_trainer,
                         self.fake_images_a,
                         self.g_B_loss_summ],
                        feed_dict={
                            self.input_a:
                                inputs['images_i'],
                            self.input_b:
                                inputs['images_j'],
                            self.learning_rate: curr_lr,
                            self.gt_a: inputs['gts_i'],
                            self.keep_rate: keep_rate_value,
                            self.is_training: is_training_value,
                            self.loss_f_weight: loss_f_weight_value,
                        }
                    )
                    writer.add_summary(summary_str, epoch * max_inter + i)

                    fake_A_temp1 = self.fake_image_pool(
                        self.num_fake_inputs, fake_A_temp, self.fake_images_A)

                    # Optimizing the D_A network
                    _, summary_str = sess.run(
                        [self.d_A_trainer, self.d_A_loss_summ],
                        feed_dict={
                            self.input_a:
                                inputs['images_i'],
                            self.input_b:
                                inputs['images_j'],
                            self.learning_rate: curr_lr,
                            self.fake_pool_A: fake_A_temp1,
                            self.keep_rate: keep_rate_value,
                            self.is_training: is_training_value,
                            self.loss_f_weight: loss_f_weight_value,
                        }
                    )
                    writer.add_summary(summary_str, epoch * max_inter + i)

                    # Optimizing the D_F network
                    _, summary_str = sess.run(
                        [self.d_F_trainer, self.d_F_loss_summ],
                        feed_dict={
                            self.input_a:
                                inputs['images_i'],
                            self.input_b:
                                inputs['images_j'],
                            self.learning_rate: curr_lr,
                            self.keep_rate: keep_rate_value,
                            self.is_training: is_training_value,
                            self.loss_f_weight: loss_f_weight_value,
                        }
                    )
                    writer.add_summary(summary_str, epoch * max_inter + i)

                    summary_str_gan, summary_str_seg, summary_str_lossf = sess.run([self.lr_gan_summ, self.lr_seg_summ, self.loss_f_weight_summ],
                             feed_dict={
                                 self.learning_rate: curr_lr,
                                 self.learning_rate_seg: curr_lr_seg,
                                 self.loss_f_weight: loss_f_weight_value,
                             })

                    writer.add_summary(summary_str_gan, epoch * max_inter + i)
                    writer.add_summary(summary_str_seg, epoch * max_inter + i)
                    writer.add_summary(summary_str_lossf, epoch * max_inter + i)

                    writer.flush()
                    self.num_fake_inputs += 1

                    if (cnt+1) % save_interval ==0:
                        saver.save(sess, os.path.join(
                            self._output_dir, "sifa"), global_step=cnt)

                sess.run(tf.assign(self.global_step, epoch + 1))

            coord.request_stop()
            coord.join(threads)
            writer.add_graph(sess.graph)


def main(log_dir, config_filename, checkpoint_dir, skip):

    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    with open(config_filename) as config_file:
        config = json.load(config_file)

    lambda_a = float(config['_LAMBDA_A']) if '_LAMBDA_A' in config else 10.0
    lambda_b = float(config['_LAMBDA_B']) if '_LAMBDA_B' in config else 10.0
    pool_size = int(config['pool_size']) if 'pool_size' in config else 50

    to_restore = False
    base_lr = float(config['base_lr']) if 'base_lr' in config else 0.0002
    max_step = int(config['max_step']) if 'max_step' in config else 200
    do_flipping = bool(config['do_flipping'])

    sifa_model = SIFA(pool_size, lambda_a, lambda_b, log_dir,
                              to_restore, base_lr, max_step,
                              checkpoint_dir, do_flipping, skip)

    sifa_model.train()


if __name__ == '__main__':
    main(log_dir='./output', config_filename='./configs/exp_01.json', checkpoint_dir='', skip=True)
