""" losses.py

    Contains loss functions used in training machine learing models throughout

    https://lars76.github.io/neural-networks/object-detection/losses-for-segmentation/

    and the paper

    https://arxiv.org/abs/1810.07842

"""
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

import timbermafia as tm
from functools import partial

from scipy.ndimage import distance_transform_edt as distance
from tensorflow.keras.losses import binary_crossentropy

import logging

log = logging.getLogger(__name__)

epsilon = tf.keras.backend.epsilon()
smooth = 1
# ----- 

def recall(y_true, y_pred):
    smooth=1
    y_pred_pos = K.clip(y_pred, 0, 1)
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.clip(y_true, 0, 1)
    tp = K.sum(y_pos * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)
    recall = (tp+smooth)/(tp+fn+smooth)
    return recall


def precision(y_true, y_pred):
    smooth=1
    y_pred_pos = K.clip(y_pred, 0, 1)
    y_pos = K.clip(y_true, 0, 1)
    y_neg = 1 - y_pos
    tp = K.sum(y_pos * y_pred_pos)
    fp = K.sum(y_neg * y_pred_pos)
    prec = (tp + smooth)/(tp+fp+smooth)
    return prec


def specificity(y_true, y_pred):
    smooth=1
    y_pred_pos = K.clip(y_pred, 0, 1)
    y_pred_neg = 1 - y_pred_pos
    y_neg = K.clip(1-y_true, 0, 1)
    tn = K.sum(y_neg * y_pred_neg)
    fp = K.sum(y_neg * y_pred_pos)
    spec = (tn + smooth)/(tn+fp+smooth)
    return spec


def npv(y_true, y_pred):
    smooth=1
    y_pred_neg = K.clip(1-y_pred, 0, 1)
    y_pos = K.clip(y_true, 0, 1)
    y_neg = 1 - y_pos
    tn = K.sum(y_neg * y_pred_neg)
    fn = K.sum(y_pos * y_pred_neg)
    npv = (tn + smooth)/(tn+fn+smooth)
    return npv


def tp(y_true, y_pred):
    smooth = 1
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pos = K.round(K.clip(y_true, 0, 1))
    tp = (K.sum(y_pos * y_pred_pos) + smooth)/ (K.sum(y_pos) + smooth)
    return tp


def tn(y_true, y_pred):
    smooth = 1
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    tn = (K.sum(y_neg * y_pred_neg) + smooth) / (K.sum(y_neg) + smooth )
    return tn

# Surface loss elements from paper
# http://proceedings.mlr.press/v102/kervadec19a/kervadec19a.pdf

def calc_dist_map(seg):
    """ Element of surface loss defined in:
        http://proceedings.mlr.press/v102/kervadec19a/kervadec19a.pdf
        https://github.com/LIVIAETS/surface-loss/issues/14#issuecomment-546342163
    """
    res = np.zeros_like(seg)
    posmask = seg.astype(np.bool)

    if posmask.any():
        negmask = ~posmask
        res = distance(negmask) * negmask - (distance(posmask) - 1) * posmask

    return res

def calc_dist_map_batch(y_true):
    """ Element of surface loss defined in:
        http://proceedings.mlr.press/v102/kervadec19a/kervadec19a.pdf
        https://github.com/LIVIAETS/surface-loss/issues/14#issuecomment-546342163
    """
    y_true_numpy = y_true.numpy()
    return np.array([calc_dist_map(y)
                     for y in y_true_numpy]).astype(np.float32)

class AlphaScheduler(tf.keras.callbacks.Callback, tm.Logged):
    """ Element of surface loss defined in:
        http://proceedings.mlr.press/v102/kervadec19a/kervadec19a.pdf
        https://github.com/LIVIAETS/surface-loss/issues/14#issuecomment-546342163

        pass alpha as: alpha = K.variable(1, dtype='float32')
        pass update_alpha as update fn
    """
    def init(self, alpha, update_fn):
        self.alpha = alpha
        self.update_fn = update_fn
    def on_epoch_end(self, epoch, logs=None):
        updated_alpha = self.update_fn(K.get_value(self.alpha))


def update_alpha(value):
    return np.clip(value - 0.01, 0.01, 1)

# tversky index

def tversky_index(y_true,
                  y_pred,
                  alpha=0.7,
                  beta=0.3,
                  eps=epsilon):
    """ The Tversky Index, a generalization of the Dice score which allows
        flexibility in the relative weighting of the importance of
        minimising FPs and FNs in training.

        Reproduces the dice coefficient and tanimoto coefficient/jaccard index
        as special cases

        Signature:

            alpha: floating to weight contribution of false negatives
            beta: float to weight contribution of false positives
            smooth: numerical smoothing factor

    """
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    # false negatives weighted by alpha
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    # false positives weighted by beta
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    return ((true_pos + eps)/
            (true_pos + alpha*false_neg + beta*false_pos + eps))


# these are all special cases of tversky index above

def dice_coefficient(y_true, y_pred, eps=epsilon):
    return tversky_index(y_true, y_pred, 0.5, 0.5, eps)

def tanimoto_coefficient(y_true, y_pred, eps=epsilon):
    return tversky_index(y_true, y_pred, 1., 1., eps)

def jaccard_index(y_true, y_pred, eps=epsilon):
    return tanimoto_coefficient(y_true, y_pred, eps)



# LOSSES derived from above components

def tversky_loss(*args, **kwargs):
    return (1-tversky_index(*args, **kwargs))

def tversky_loss_tunable(alpha=0.9, beta=0.1):
    def _tversky_loss(y_true, y_pred):
        y_true_pos = K.flatten(y_true)
        y_pred_pos = K.flatten(y_pred)
        true_pos = K.sum(y_true_pos * y_pred_pos)
        # false negatives weighted by alpha
        false_neg = K.sum(y_true_pos * (1-y_pred_pos))
        # false positives weighted by beta
        false_pos = K.sum((1-y_true_pos)*y_pred_pos)
        return ((true_pos + eps)/
                (true_pos + alpha*false_neg + beta*false_pos + eps))
    return _tversky_loss

def focal_tversky_loss(y_true, y_pred, *args, gamma=4./3, **kwargs):
    """ https://arxiv.org/pdf/1810.07842.pdf """
    ti = tversky_index(y_true, y_pred, *args, **kwargs)
    return K.pow((1-ti), 1/gamma)

def dice_coeff_loss(y_true, y_pred, eps=epsilon):
    return 1. - dice_coefficient(y_true, y_pred, eps)

def bce_dice_loss(y_true, y_pred, eps=epsilon):
    loss = binary_crossentropy(y_true, y_pred) + dice_coeff_loss(y_true, y_pred, eps)
    return loss

def class_weighted_pixelwise_crossentropy(sample_weights):
    def _cw_px_ce(target, output):
        output = tf.clip_by_value(output, 10e-8, 1.-10e-8)
        return -tf.reduce_sum(target * sample_weights * tf.math.log(output))
    return _cw_px_ce

def binary_crossentropy_from_logits(y_true, y_pred):
     return binary_crossentropy(y_true, y_pred, from_logits=True)

def weighted_binary_crossentropy(pos_wgt=10, eps=epsilon):
    """ version with a fixed weight for the whole dataset """
    def _wbce(y_true, y_pred):
        return - tf.reduce_mean(
            pos_wgt * y_true * K.log(y_pred + eps) +
            (1. - y_true) * K.log(1. - y_pred + eps)
        )

    return _wbce


def bbce_adaptive(y_true, y_pred, eps=tf.keras.backend.epsilon()):
    """ version that precalculates a weight for each class
        https://arxiv.org/pdf/1707.03237.pdf (pretty sure they mean r_n instead of p_n)
        to balance class freqs (see 'w_c' in )
        https://arxiv.org/pdf/1505.04597.pdf

        normalises weights st pos_wgt + neg_wgt  = 1.
        see https://lars76.github.io/neural-networks/object-detection/losses-for-segmentation/
        balanced cross entropy

        this can sometimes go negative with current log setup?
    """
    #N = tf.size(y_true, out_type=tf.dtypes.float32)
    pos_y_sum = tf.reduce_sum(y_true)
    neg_y_sum = tf.reduce_sum(1. - y_true)#y_true.size - pos_y_sum#N - pos_y_sum#
    beta = (neg_y_sum + eps) / y_true.size
    #log.debug("beta",beta)
    return - tf.reduce_mean(
        beta * y_true * K.log(y_pred + eps) +
        (1. - y_true) * (1. - beta) * K.log(1. - y_pred + eps)
    )

def wbce_adaptive(y_true, y_pred, eps=tf.keras.backend.epsilon()):
    """ version that precalculates a weight map for each image
        https://arxiv.org/pdf/1707.03237.pdf (pretty sure they mean r_n instead of p_n)
        to balance class freqs (see 'w_c' in )
        https://arxiv.org/pdf/1505.04597.pdf
    """
    #N = tf.size(y_true, out_type=tf.dtypes.float32)
    pos_y_sum = tf.reduce_sum(y_true)
    neg_y_sum = tf.reduce_sum(1. - y_true)#y_true.size - pos_y_sum#N - pos_y_sum#
    pos_wgt = (neg_y_sum + eps)/(pos_y_sum + eps)#(N - y_sum + eps)/(y_sum + eps)
    #log.debug("pos_wgt", pos_wgt)
    return - tf.reduce_mean(
        pos_wgt * y_true * K.log(y_pred + eps) +
        (1. - y_true) * K.log(1. - y_pred + eps)
    )

def wbce_adaptive_dice_loss(y_true, y_pred, lambda_=.1, eps=tf.keras.backend.epsilon()):
    loss = lambda_ * wbce_adaptive(y_true, y_pred, eps=eps) + dice_coeff_loss(y_true, y_pred, eps=eps)
    return loss

def weighted_wbce_adaptive_dice_loss(lambda_, eps=tf.keras.backend.epsilon()):
    """ lambda * weighted bce loss + dice coeff"""
    f = partial(wbce_adaptive_dice_loss, lambda_=lambda_, eps=eps)
    return f

def surface_loss(y_true, y_pred):
    y_true_dist_map = tf.py_function(func=calc_dist_map_batch,
                                     inp=[y_true],
                                     Tout=tf.float32)
    multipled = y_pred * y_true_dist_map
    return K.mean(multipled)


#
# Lovasz-Softmax and Jaccard hinge loss in Tensorflow
# Maxim Berman 2018 ESAT-PSI KU Leuven (MIT License)#
#
# https://github.com/bermanmaxim/LovaszSoftmax/blob/master/tensorflow/lovasz_losses_tf.py
#

import tensorflow as tf
import numpy as np


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    gts = tf.reduce_sum(gt_sorted)
    intersection = gts - tf.cumsum(gt_sorted)
    union = gts + tf.cumsum(1. - gt_sorted)
    jaccard = 1. - intersection / union
    jaccard = tf.concat((jaccard[0:1], jaccard[1:] - jaccard[:-1]), 0)
    return jaccard


# --------------------------- BINARY LOSSES ---------------------------

def lovasz_hinge_loss(y_true, y_pred):
    return lovasz_hinge(y_pred, y_true)

def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        def treat_image(log_lab):
            log, lab = log_lab
            log, lab = tf.expand_dims(log, 0), tf.expand_dims(lab, 0)
            log, lab = flatten_binary_scores(log, lab, ignore)
            return lovasz_hinge_flat(log, lab)
        losses = tf.map_fn(treat_image, (logits, labels), dtype=tf.float32)
        loss = tf.reduce_mean(losses)
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """

    def compute_loss():
        labelsf = tf.cast(labels, logits.dtype)
        signs = 2. * labelsf - 1.
        errors = 1. - logits * tf.stop_gradient(signs)
        errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0], name="descending_sort")
        gt_sorted = tf.gather(labelsf, perm)
        grad = lovasz_grad(gt_sorted)
        loss = tf.tensordot(tf.nn.relu(errors_sorted), tf.stop_gradient(grad), 1, name="loss_non_void")
        return loss

    # deal with the void prediction case (only void pixels)
    loss = tf.compat.v1.cond(
        tf.equal(tf.shape(logits)[0], 0),
        lambda: tf.reduce_sum(logits) * 0.,
        compute_loss,
        strict=True,
        name="loss"
    )

    #loss = tf.cond(tf.equal(tf.shape(logits)[0], 0),
    #               lambda: tf.reduce_sum(logits) * 0.,
    #              compute_loss,
    #               strict=True,
    #               name="loss"
    #               )
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = tf.reshape(scores, (-1,))
    labels = tf.reshape(labels, (-1,))
    if ignore is None:
        return scores, labels
    valid = tf.not_equal(labels, ignore)
    vscores = tf.boolean_mask(scores, valid, name='valid_scores')
    vlabels = tf.boolean_mask(labels, valid, name='valid_labels')
    return vscores, vlabels


# --------------------------- MULTICLASS LOSSES ---------------------------

# verbatim from attn unet repo: https://github.com/nabsabraham/focal-tversky-unet

def dsc(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score

def tversky(y_true, y_pred):
    smooth=1
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true,y_pred)

def focal_tversky(y_true,y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return K.pow((1-pt_1), gamma)
