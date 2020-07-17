
import tensorflow as tf


class BiasLayer(tf.keras.layers.Layer):
    def __init__(self, n_classes, **kwargs):
        self.n_classes = n_classes
        super(BiasLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.bias = self.add_weight(shape=self.n_classes - 1, initializer='zeros', dtype=tf.float32, name='bias_logits')

    def call(self, inputs, **kwargs):
        expander = tf.ones_like(self.bias)
        inputs = inputs * expander
        return tf.add(inputs, self.bias)

    def get_config(self):
        config = {'n_classes': self.n_classes}
        base_config = super(BiasLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


