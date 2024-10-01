# Script for U-Net 
# Important imports
import tensorflow as tf
from tensorflow.keras.layers import MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation
from tensorflow.keras.models import Model


# U-net Class
class Unet(tf.keras.Model):
    """
    U-Net class 
    Use to build the U-Net architecture 

    :param input_shape: Shape of input tensor 

    return:
        model: U-net model
    """

    def __init__(self, encoder, decoder, conv):
        super(Unet, self).__init__()

        self.encoder_block = encoder
        self.decoder_block = decoder
        self.conv_block = conv


    def build_unet(self, input_shape):

        # Input 
        self.inputs = Input(input_shape)

        # Encoder Blocks
        self.s1, self.p1 = self.encoder_block(self.inputs, 64)
        self.s2, self.p2 = self.encoder_block(self.p1, 128)
        self.s3, self.p3 = self.encoder_block(self.p2, 256)
        self.s4, self.p4 = self.encoder_block(self.p3, 512)

        # Convolution Block
        self.b1 = self.conv_block(self.p4, 1024)

        # Decoder Block
        self.d1 = self.decoder_block(self.b1, self.s4, 512)
        self.d2 = self.decoder_block(self.d1, self.s3, 256)
        self.d3 = self.decoder_block(self.d2, self.s2, 128)
        self.d4 = self.decoder_block(self.d3, self.s1, 64)

        self.outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(self.d4)

        self.model = Model(inputs=self.inputs, outputs=self.outputs, name="UNET")

        return self.model 
    

    
