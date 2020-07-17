import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np


def task_importance_weights(label_array):
    unique = np.unique(label_array)
    total = len(label_array)
    m = np.zeros(unique.shape[0])

    for i, t in enumerate(np.arange(np.min(unique), np.max(unique))):
        s_k = len(label_array[label_array > t])
        m_k = max(s_k, total - s_k)
        m[i] = np.sqrt(np.float(m_k))

    imp = m / np.max(m)
    imp_tensor = tf.convert_to_tensor(imp[0:unique.shape[0] - 1], dtype=float)

    return imp_tensor


def loss(n_classes, imp_w=None):
    def loss_fn(true_logits, pred_logits):
        imp = imp_w if imp_w is not None else tf.ones(n_classes - 1, dtype=float)
        val = (-K.sum(
                (K.log(K.sigmoid(pred_logits)) * true_logits
                 + (K.log(K.sigmoid(pred_logits)) - pred_logits)
                 * (1 - true_logits)) * imp,
               axis=1))
        return K.mean(val)

    return loss_fn
