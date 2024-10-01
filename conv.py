# Script for Convolution Block function
import numpy as np
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation


def conv_block(inputs: np.array, num_filters: int):
    """
    Convolutional Block
    :param inputs: input tensor
    :param num_filters: number of filters
    :return: tensor
    """

    # First layer
    x = Conv2D(num_filters, 3, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Second layer
    x = Conv2D(num_filters, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x