import tensorflow as tf


def cycle_consistency_loss(real_images, generated_images):
    """
    Compute the cycle consistency loss.
    """
    return tf.reduce_mean(tf.abs(real_images - generated_images))


def lsgan_loss_generator(prob_fake_is_real):
    """
    Computes the LS-GAN loss as minimized by the generator.
    """
    return tf.reduce_mean(tf.squared_difference(prob_fake_is_real, 1))


def lsgan_loss_discriminator(prob_real_is_real, prob_fake_is_real):
    """
    Computes the LS-GAN loss as minimized by the discriminator.
    """
    return (tf.reduce_mean(tf.squared_difference(prob_real_is_real, 1)) +
            tf.reduce_mean(tf.squared_difference(prob_fake_is_real, 0))) * 0.5


def _softmax_weighted_loss(logits, gt):
    """
    Calculate weighted cross-entropy loss.
    """
    softmaxpred = tf.nn.softmax(logits)
    for i in xrange(5):
        gti = gt[:,:,:,i]
        predi = softmaxpred[:,:,:,i]
        weighted = 1-(tf.reduce_sum(gti) / tf.reduce_sum(gt))
        if i == 0:
            raw_loss = -1.0 * weighted * gti * tf.log(tf.clip_by_value(predi, 0.005, 1))
        else:
            raw_loss += -1.0 * weighted * gti * tf.log(tf.clip_by_value(predi, 0.005, 1))

    loss = tf.reduce_mean(raw_loss)

    return loss


def _dice_loss_fun(logits, gt):
    """
    Calculate dice loss.
    """
    dice = 0
    eps = 1e-7
    softmaxpred = tf.nn.softmax(logits)
    for i in xrange(5):
        inse = tf.reduce_sum(softmaxpred[:, :, :, i]*gt[:, :, :, i])
        l = tf.reduce_sum(softmaxpred[:, :, :, i]*softmaxpred[:, :, :, i])
        r = tf.reduce_sum(gt[:, :, :, i])
        dice += 2.0 * inse/(l+r+eps)

    return 1 - 1.0 * dice / 5


def task_loss(prediction, gt):
    """
    Calculate task loss, which consists of the weighted cross entropy loss and dice loss
    """
    ce_loss = _softmax_weighted_loss(prediction, gt)
    dice_loss = _dice_loss_fun(prediction, gt)

    return ce_loss, dice_loss