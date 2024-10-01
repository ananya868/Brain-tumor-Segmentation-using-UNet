# Script for encoder block function 
import numpy as np
from conv import conv_block
from tensorflow.keras.layers import MaxPool2D, Conv2DTranspose, Concatenate, Input


def encoder_block(inputs: np.array, num_filters: int):
    """
    Encoder block
    :param inputs: input tensor
    :param num_filters: number of filters
    :return: tensor, keras.layer
    """

    # Use conv-block 
    x = conv_block(inputs, num_filters)
    # Use Max pool layer
    p = MaxPool2D((2, 2))(x)

    return x, p