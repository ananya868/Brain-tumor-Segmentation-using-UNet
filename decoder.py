# Script for decoder block function 
import numpy as np
from tensorflow.keras.layers import MaxPool2D, Conv2DTranspose, Concatenate, Input
from conv import conv_block


def decoder_block(inputs: np.array, skip_features: np.array, num_filters: int):
    """
    Decoder block
    :param inputs: input tensor
    :param skip_features: np.array
    :param num_filters: int
    :return: tensor
    """
    
    # Conv 2D
    x = Conv2DTranspose(num_filters, 2, strides=2, padding="same")(inputs)
    x = Concatenate()([x, skip_features])
    # Use Conv-block
    x = conv_block(x, num_filters)
 
    return x