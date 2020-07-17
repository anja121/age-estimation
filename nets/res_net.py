

from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, AveragePooling2D, BatchNormalization, Conv2D, \
                                           MaxPooling2D, Input, Dense, GlobalAveragePooling2D
from nets.bias_layer import BiasLayer

"""
Implementation of ResNet CORAL nn architecture(s).
ResNet variations ResNet18, ResNet34, ResNet50 etc can be set by "net_code" in ResNet_CORAL function.
All codes can be seen in "RESNET_CODES" dict.
"""

RESNET_CODES = {
    18: ([2, 2, 2, 2], False),
    34: ([3, 4, 6, 3], False),
    50: ([3, 4, 6, 3], True),
    101: ([3, 4, 23, 3], True),
    152: ([3, 8, 36, 3], True)
}


def identity_block_bottleneck(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(
      filters2, kernel_size, padding='same', name=conv_name_base + '2b')(
          x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block_bottleneck(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    x = Conv2D(
        filters1, (1, 1), strides=strides, name=conv_name_base + '2a')(
        input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(
        filters2, kernel_size, padding='same', name=conv_name_base + '2b')(
        x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(
        filters3, (1, 1), strides=strides, name=conv_name_base + '1')(
        input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def identity_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.ZeroPadding2D(padding=(1, 1))(input_tensor)
    x = Conv2D(filters1, kernel_size, name=conv_name_base + '2a')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):

    filters1, filters2 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.ZeroPadding2D(padding=(1, 1))(input_tensor)
    x = Conv2D(filters1, kernel_size, strides=strides, name=conv_name_base + '2a')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    shortcut = Conv2D(
      filters2, (1, 1), strides=strides, name=conv_name_base + '1')(
          input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def ResNet_TOP(input_layer, repetitions, bottleneck=False):
    b1, b2, b3, b4 = repetitions

    if bottleneck:
        conv = conv_block_bottleneck
        identity = identity_block_bottleneck
        filters = [64, 64, 256]
    else:
        conv = conv_block
        identity = identity_block
        filters = [64, 64]

    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    # BLOCK 0: input block
    x = Conv2D(
      64, (7, 7), strides=(2, 2), padding='same', name='conv1')(input_layer)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # BLOCK 1
    x = conv(x, 3, filters, stage=2, block=str(0), strides=(1, 1))
    for i in range(b1-1):
        x = identity(x, 3, filters, stage=2, block=str(i+1))

    # BLOCK 2
    x = conv(x, 3, [f * 2 for f in filters], stage=3, block=str(0))
    for i in range(b2-1):
        x = identity(x, 3, [f * 2 for f in filters], stage=3, block=str(i+1))

    # BLOCK 3
    x = conv(x, 3, [f * 4 for f in filters], stage=4, block=str(0))
    for i in range(b3 - 1):
        x = identity(x, 3, [f * 4 for f in filters], stage=4, block=str(i+1))

    # BLOCK 4
    x = conv(x, 3, [f * 8 for f in filters], stage=5, block=str(0))
    for i in range(b4 - 1):
        x = identity(x, 3, [f * 8 for f in filters], stage=5, block=str(i+1))

    x = AveragePooling2D((7, 7), name='avg_pool')(x)

    return x


def ResNet_CORAL(input_shape, n_classes, net_code=18):
    input_layer = Input(shape=input_shape)
    net_conf = RESNET_CODES.get(net_code)
    resnet = ResNet_TOP(input_layer, net_conf[0], bottleneck=net_conf[1])

    x = GlobalAveragePooling2D()(resnet)

    # Logits Block
    logits = Dense(1)(x)
    biased_logits = BiasLayer(n_classes=n_classes, name="logits")(logits)
    probas = Activation('sigmoid', name="probas")(biased_logits)

    return Model(inputs=[input_layer], outputs=[probas, biased_logits])



