import numpy as np
import tensorflow as tf
import pandas as pd
import random

AUTOTUNE = tf.data.experimental.AUTOTUNE


class MorphDataset:
    """
    Load and prepare morph image data with extended labels

    data_df: pandas DataFrame; example head
    index,  ID,     Path,                   Gender, Race, Age, scaled
    0,     134083, Album2/134083_05M41.JPG,   M,    B,    41,    25

    img_dir: string; path to dataset folder
    img_size: (img_width, img_height)
    channels: img channels
    seed: seed
    """
    def __init__(self, data_df, img_dir, img_size=(128, 128), channels=3, seed=None):

        self.data_df = data_df
        self.img_dir = img_dir
        self.img_size = img_size
        self.channels = channels
        if seed is not None:
            self.seed = seed
        else:
            self.seed = random.randint(0, 1000)
        self.classes = sorted(np.unique(self.data_df["scaled"].values))

        self.img_paths = self.get_image_paths()
        self.labels = self.get_labels()
        self.label_array = self.data_df["scaled"].values
        self.size = len(self.data_df)

    def get_image_paths(self):
        return self.data_df["Path"].apply(lambda x: self.img_dir + x).values

    def get_extended_label(self, label):
        levels = [1] * label + [0] * (len(self.classes)-1-label)
        return levels

    def get_labels(self):
        return list(self.data_df["scaled"].apply(lambda x: self.get_extended_label(x)).values)

    def read_image(self, img_path):
        raw_img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(raw_img, channels=self.channels)
        return img

    def resize_image(self, img):
        return tf.image.resize(img, self.img_size)

    def normalize_image(self, img):
        return tf.cast(img, tf.float32) / 255.0

    def batch(self, batch_size, shuffle=False, shuffle_buffer=None):
        paths = tf.data.Dataset.from_tensor_slices(self.img_paths)

        labels = tf.data.Dataset.from_tensor_slices(np.array(self.labels))

        images = paths.map(self.read_image, num_parallel_calls=AUTOTUNE)
        images = images.map(self.resize_image, num_parallel_calls=AUTOTUNE)
        images = images.map(self.normalize_image, num_parallel_calls=AUTOTUNE)

        data = tf.data.Dataset.zip((images, labels))
        data = data.repeat()

        if shuffle:
            buffer = shuffle_buffer if shuffle_buffer is not None else self.size
            data = data.prefetch(AUTOTUNE)\
                       .shuffle(buffer)\
                       .batch(batch_size)
        else:
            data = data.batch(batch_size)\
                       .prefetch(AUTOTUNE)

        return data
