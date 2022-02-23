""" metrics.py

    Contains metrics (loss functions) used in training machine learing models throughout

    https://lars76.github.io/neural-networks/object-detection/losses-for-segmentation/

    and the paper

    https://arxiv.org/abs/1810.07842

"""

# tests
#precision(np.array((1, 1, 0)), np.array((1., 1., 1.)))# - 0.6666 < eps
#recall(np.array((1, 1, 0)), np.array((1., 0., 0.)))# - 0.5000 < eps
#specificity(np.array((1, 0, 0)), np.array((1., 0., 1.)))# - 0.5000 < eps
#npv(np.array((1, 0, 0)), np.array((0, 0, 0)))# - 0.6666 < eps

import numpy as np
import sys

#epsilon = sys.float_info.epsilon
epsilon = 1e-9

def recall(y_true, y_pred, eps=epsilon):
    """
    The Recall statistic, TP/TP+FN
    
    Parameters
    ----------
    y_true: :obj:`numpy.ndarray`
        An array of ground truth binary label values in {0, 1}
    y_pred: :obj:`numpy.ndarray`
        An array of predicted probabilities of belonging to these labels
    eps: float, optional
        Tolerance parameter to avoid division by zero
        
    Returns
    -------
    float:
        The recall value in [0, 1]
    """
    y_true_flat = np.clip(y_true.flatten(), 0, 1)
    y_pred_flat = np.clip(y_pred.flatten(), 0, 1)
    # ignore elements where nan values appear in case these are masked out
    no_nan = ~np.isnan(y_true_flat) & ~np.isnan(y_pred_flat)
    y_true_pos = y_true_flat[no_nan]
    y_pred_pos = y_pred_flat[no_nan]
    y_pred_neg = 1 - y_pred_pos
    tp = np.sum(y_true_pos * y_pred_pos)
    fn = np.sum(y_true_pos * y_pred_neg)
    recall = (tp+eps)/(tp+fn+eps)
    return recall


def precision(y_true, y_pred, eps=epsilon):
    """
    The Precision statistic, TP/TP+FP
    
    Parameters
    ----------
    y_true: :obj:`numpy.ndarray`
        An array of ground truth binary label values in {0, 1}
    y_pred: :obj:`numpy.ndarray`
        An array of predicted probabilities of belonging to these labels
    eps: float, optional
        Tolerance parameter to avoid division by zero
        
    Returns
    -------
    float:
        The precision value in [0, 1]
    """
    y_true_flat = np.clip(y_true.flatten(), 0, 1)
    y_pred_flat = np.clip(y_pred.flatten(), 0, 1)
    # ignore elements where nan values appear in case these are masked out
    no_nan = ~np.isnan(y_true_flat) & ~np.isnan(y_pred_flat)
    y_true_pos = y_true_flat[no_nan]
    y_pred_pos = y_pred_flat[no_nan]
    y_true_neg = 1 - y_true_pos
    tp = np.sum(y_true_pos * y_pred_pos)
    fp = np.sum(y_true_neg * y_pred_pos)
    prec = (tp + eps)/(tp+fp+eps)
    return prec


def specificity(y_true, y_pred, eps=epsilon):
    """
    The Specificity statistic, TN/TN+FP
    
    Parameters
    ----------
    y_true: :obj:`numpy.ndarray`
        An array of ground truth binary label values in {0, 1}
    y_pred: :obj:`numpy.ndarray`
        An array of predicted probabilities of belonging to these labels
    eps: float, optional
        Tolerance parameter to avoid division by zero
        
    Returns
    -------
    float:
        The specificity value in [0, 1]
    """
    y_true_flat = np.clip(y_true.flatten(), 0, 1)
    y_pred_flat = np.clip(y_pred.flatten(), 0, 1)
    # ignore elements where nan values appear in case these are masked out
    no_nan = ~np.isnan(y_true_flat) & ~np.isnan(y_pred_flat)
    y_true_pos = y_true_flat[no_nan]
    y_pred_pos = y_pred_flat[no_nan]
    y_pred_neg = 1 - y_pred_pos
    y_true_neg = 1 - y_true_pos
    tn = np.sum(y_true_neg * y_pred_neg)
    fp = np.sum(y_true_neg * y_pred_pos)
    spec = (tn + eps)/(tn+fp+eps)
    return spec


def npv(y_true, y_pred, eps=epsilon):
    """
    The NPV statistic, TN/TN+FN
    
    Parameters
    ----------
    y_true: :obj:`numpy.ndarray`
        An array of ground truth binary label values in {0, 1}
    y_pred: :obj:`numpy.ndarray`
        An array of predicted probabilities of belonging to these labels
    eps: float, optional
        Tolerance parameter to avoid division by zero
        
    Returns
    -------
    float:
        The NPV value in [0, 1]
    """
    y_true_flat = np.clip(y_true.flatten(), 0, 1)
    y_pred_flat = np.clip(y_pred.flatten(), 0, 1)
    # ignore elements where nan values appear in case these are masked out
    no_nan = ~np.isnan(y_true_flat) & ~np.isnan(y_pred_flat)
    y_true_pos = y_true_flat[no_nan]
    y_pred_pos = y_pred_flat[no_nan]
    y_pred_neg = 1 - y_pred_pos
    y_true_neg = 1 - y_true_pos
    tn = np.sum(y_true_neg * y_pred_neg)
    fn = np.sum(y_true_pos * y_pred_neg)
    npv = (tn + eps)/(tn+fn+eps)
    return npv


def tversky_index(y_true,
                  y_pred,
                  alpha=0.7,
                  beta=0.3,
                  eps=epsilon):
    """ 
    The Tversky Index metric for segmentation
    
    A generalization of the Dice score which allows flexibility in the 
    relative weighting of the importance of FPs and FNs

    Reproduces the dice coefficient and tanimoto coefficient/jaccard index
    as special cases.

    Parameters
    ----------
    alpha: 
        weight for contribution of false negatives
    beta: float
        weight for contribution of false positives
    eps: float
        numerical smoothing factor
    
    Returns
    -------
    float:
        The tversky index value in [0, 1]
    """
    y_true_flat = np.clip(y_true.flatten(), 0, 1)
    y_pred_flat = np.clip(y_pred.flatten(), 0, 1)
    # ignore elements where nan values appear in case these are masked out
    no_nan = ~np.isnan(y_true_flat) & ~np.isnan(y_pred_flat)
    y_true_pos = y_true_flat[no_nan]
    y_pred_pos = y_pred_flat[no_nan]
    true_pos = np.sum(y_true_pos * y_pred_pos)
    # false negatives weighted by alpha
    false_neg = np.sum(y_true_pos * (1-y_pred_pos))
    # false positives weighted by beta
    false_pos = np.sum((1-y_true_pos)*y_pred_pos)
    return ((true_pos + eps)/
            (true_pos + alpha*false_neg + beta*false_pos + eps))


# these are all special cases of tversky index above

def dice_coefficient(y_true, y_pred, eps=epsilon):
    return tversky_index(y_true, y_pred, 0.5, 0.5, eps)

def tanimoto_coefficient(y_true, y_pred, eps=epsilon):
    return tversky_index(y_true, y_pred, 1., 1., eps)

def jaccard_index(y_true, y_pred, eps=epsilon):
    return tanimoto_coefficient(y_true, y_pred, eps)