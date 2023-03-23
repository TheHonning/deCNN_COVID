import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import *
import matplotlib.pyplot as plt

class SarsCovCNN(tf.keras.Model):

    @tf.function
    def __init__(self, img_height:int = 300, img_width:int = 300) -> None:
        super(SarsCovCNN, self).__init__()

        input_shape = (img_height, img_width, 3)
        kernel_sizes = [(3 * 3), (6 * 6), (9 * 9)]
        pooling_factors = [2, 4 ,6, 8, 10]

        self.C1 = Conv2D(32, kernel_sizes[0], padding='same', input_shape = input_shape)
        self.B1 = BatchNormalization()
        self.A1 = Activation('relu')
        self.P1 = MaxPooling2D(pooling_factors[0], padding='same')
