import theano.tensor as T
import numpy as np
from theano import config

from lasagne.objectives import squared_error as squared_error_lasagne

_FLOATX = config.floatX
_EPSILON = 10e-8


def jaccard(y_pred, y_true, n_classes, one_hot=False):

    assert (y_pred.ndim == 2) or (y_pred.ndim == 1)

    # y_pred to indices
    if y_pred.ndim == 2:
        y_pred = T.argmax(y_pred, axis=1)

    if one_hot:
        y_true = T.argmax(y_true, axis=1)

    # Compute confusion matrix
    cm = T.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            cm = T.set_subtensor(
                cm[i, j], T.sum(T.eq(y_pred, i) * T.eq(y_true, j)))

    # Compute Jaccard Index
    TP_perclass = T.cast(cm.diagonal(), _FLOATX)
    FP_perclass = cm.sum(1) - TP_perclass
    FN_perclass = cm.sum(0) - TP_perclass

    num = TP_perclass
    denom = TP_perclass + FP_perclass + FN_perclass

    return T.stack([num, denom], axis=0)


def accuracy(y_pred, y_true, void_labels, one_hot=False):

    assert (y_pred.ndim == 2) or (y_pred.ndim == 1)

    # y_pred to indices
    if y_pred.ndim == 2:
        y_pred = T.argmax(y_pred, axis=1)

    if one_hot:
        y_true = T.argmax(y_true, axis=1)

    # Compute accuracy
    acc = T.eq(y_pred, y_true).astype(_FLOATX)

    # Create mask
    mask = T.ones_like(y_true, dtype=_FLOATX)
    for el in void_labels:
        indices = T.eq(y_true, el).nonzero()
        if any(indices):
            mask = T.set_subtensor(mask[indices], 0.)

    # Apply mask
    acc *= mask
    acc = T.sum(acc) / T.sum(mask)

    return acc


def crossentropy(y_pred, y_true, void_labels, one_hot=False):
    # Clip predictions
    y_pred = T.clip(y_pred, _EPSILON, 1.0 - _EPSILON)

    if one_hot:
        y_true = T.argmax(y_true, axis=1)

    # Create mask
    mask = T.ones_like(y_true, dtype=_FLOATX)
    for el in void_labels:
        mask = T.set_subtensor(mask[T.eq(y_true, el).nonzero()], 0.)

    # Modify y_true temporarily
    y_true_tmp = y_true * mask
    y_true_tmp = y_true_tmp.astype('int32')

    # Compute cross-entropy
    loss = T.nnet.categorical_crossentropy(y_pred, y_true_tmp)

    # Compute masked mean loss
    loss *= mask
    loss = T.sum(loss) / T.sum(mask)

    return loss


def binary_crossentropy(y_pred, y_true):
    # Clip predictions to avoid numerical instability
    y_pred = T.clip(y_pred, _EPSILON, 1.0 - _EPSILON)

    loss = T.nnet.binary_crossentropy(y_pred, y_true)

    return loss.mean()


def entropy(y_pred):
    # Clip predictions to avoid numerical instability
    y_pred = T.clip(y_pred, _EPSILON, 1.0 - _EPSILON)

    ent = - T.sum(y_pred * T.log(y_pred), axis=1)

    return ent.mean()


def squared_error_h(y_pred, y_true):

    coef = np.linspace(_EPSILON, 1, len(y_pred)+1)[:-1]

    error_list = [((a_i - b_i)**2).mean() for
                  a_i, b_i in zip(y_pred, y_true)]
    error_list = error_list * coef

    return sum(error_list)


def squared_error(y_pred, y_true, void):

    if isinstance(void, int):
        loss_aux = squared_error_lasagne(y_pred, y_true[:, :void, :, :]).mean(axis=1)
        mask = y_true[:, :void, :, :].sum(axis=1)
    else:  # assumes b,c,0,1
        loss_aux = squared_error_lasagne(y_pred, y_true).mean(axis=1)
        mask = T.neq(y_true.sum(1), sum(void))

    loss_aux = loss_aux * mask
    loss = loss_aux.sum()/mask.sum()

    return loss
