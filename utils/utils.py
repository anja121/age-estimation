from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

import os
import numpy as np

from nets.vgg_net import VGG_CORAL
from nets.res_net import ResNet_CORAL
from utils.loss import loss, task_importance_weights
from utils.metrics import mae_logits, rmse_logits
from tensorflow.keras.models import load_model
from nets.bias_layer import BiasLayer

import json
import sys


# UTILS


def read_config_file(path):

    try:
        with open(path) as config_file:
            conf = json.load(config_file)
            return conf
    except IOError:
        sys.exit()


# TRAIN UTILS

def get_callbacks(conf, folder_name):
    callbacks = []

    if conf["checkpoint"]:
        checkpoint_dir = "checkpoints/" + folder_name

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        checkpoint_callback = ModelCheckpoint(checkpoint_dir + "/age-estimation" +
                                              '-{epoch:03d}-{loss:03f}.h5',
                                              verbose=1,
                                              monitor='loss',
                                              save_best_only=True,
                                              mode='auto')
        callbacks.append(checkpoint_callback)

    if conf["logs"]:
        logdir = "logs/" + folder_name

        tensorboard_callback = TensorBoard(log_dir=logdir)
        callbacks.append(tensorboard_callback)

    return callbacks


def instantiate_model(conf, n_classes):
    if conf["arch_type"] == "resnet":
        model = ResNet_CORAL((conf["img_size"], conf["img_size"], 3), n_classes, conf["arch_subtype"])
    elif conf.arch_type == "vgg":
        model = VGG_CORAL((conf["img_size"], conf["img_size"], 3), n_classes)
    else:
        raise Exception("Desired model is not defined!")
    return model


def get_model(conf, n_classes, label_array=None):
    model = instantiate_model(conf, n_classes)
    imp_weights = None

    if label_array is not None:
        imp_weights = task_importance_weights(label_array)
        imp_weights = imp_weights[0:n_classes-1]

    loss_func = {"logits": loss(n_classes=n_classes, imp_w=imp_weights)}
    metrics_func = {"probas": [mae_logits, rmse_logits]}
    model.compile(optimizer=Adam(), loss=loss_func, metrics=metrics_func)

    return model


# PREDICT UTILS


def load_custom_model(model_path, n_classes):
    model = load_model(model_path,
                       custom_objects={'loss_fn': loss(n_classes=n_classes),
                                       'BiasLayer': BiasLayer(n_classes=n_classes),
                                       "mae_logits": mae_logits,
                                       "rmse_logits": rmse_logits})
    model.summary()
    return model


def get_thresholded_probas(threshold, probas):
    return [1 if proba > threshold else 0 for proba in probas[0]]


def get_result_value(threshold, probas, offset_val):
    return np.sum(get_thresholded_probas(threshold, probas)) + offset_val


def save_log_file(name, threshold, probas, logits, offset):
    log_dict = {"raw_probas": probas.tolist(),
                "raw_logits": logits.tolist(),
                "threshold": threshold,
                "thresholded_probas": get_thresholded_probas(threshold, probas),
                "result_value": int(get_result_value(threshold, probas, offset))}

    with open(name + ".json", "w") as log_file:
        json.dump(log_dict, log_file)
