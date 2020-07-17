import tensorflow as tf
from tensorflow.keras import backend as K


def get_values(true_logits, pred_probas):
    pred_levels = tf.cast((pred_probas > 0.5), tf.float32)
    pred = K.sum(pred_levels, axis=1)
    true = K.sum(true_logits, axis=1)
    return true, pred


def mae_logits(true_logits, pred_probas):
    true, pred = get_values(true_logits, pred_probas)
    return K.mean(K.abs(true - pred))


def rmse_logits(true_logits, pred_probas):
    true, pred = get_values(true_logits, pred_probas)
    return K.sqrt(K.mean(K.square(true - pred)))

