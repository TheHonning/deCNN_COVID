import tensorflow as tf
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


class SarsCovCNN(tf.keras.Model):

    @tf.function
    def __init__(self, img_height: int = 300, img_width: int = 300) -> None:
        super(SarsCovCNN, self).__init__()

        input_shape = (img_height, img_width, 3)
        kernel_sizes = [(3 * 3), (6 * 6), (9 * 9)]
        pooling_factors = [2, 4, 6, 8, 10]

        self.C1 = Conv2D(32, kernel_sizes[0], padding='same',
                         input_shape=input_shape)
        self.B1 = BatchNormalization()
        self.A1 = Activation('relu')
        self.P1 = MaxPooling2D(pooling_factors[0], padding='same')


class VGG16(tf.keras.Model):
    def __init__(self) -> None:
        super().__init__()
        self.conv1_1 = Conv2D(64, (3, 3), padding="same", activation="relu",
                              input_shape=(32, 224, 224, 3))

        self.conv1_2 = Conv2D(64, (3, 3), padding="same", activation="relu")
        
        self.pool1 = MaxPooling2D(strides=(2, 2))
        self.conv2_1 = Conv2D(128, (3, 3), padding="same", activation="relu")

        self.conv2_2 = Conv2D(128, (3, 3), padding="same", activation="relu")

        self.pool2 = MaxPooling2D(strides=(2, 2))
        self.conv3_1 = Conv2D(256, (3, 3), padding="same", activation="relu")

        self.conv3_2 = Conv2D(256, (3, 3), padding="same", activation="relu")

        self.conv3_3 = Conv2D(256, (3, 3), padding="same", activation="relu")

        self.pool3 = MaxPooling2D(strides=(2, 2))
        self.conv4_1 = Conv2D(512, (3, 3), padding="same", activation="relu")

        self.conv4_2 = Conv2D(512, (3, 3), padding="same", activation="relu")

        self.conv4_3 = Conv2D(512, (3, 3), padding="same", activation="relu")

        self.pool4 = MaxPooling2D(strides=(2, 2))
        self.conv5_1 = Conv2D(512, (3, 3), padding="same", activation="relu")

        self.conv5_2 = Conv2D(512, (3, 3), padding="same", activation="relu")

        self.conv5_3 = Conv2D(512, (3, 3), padding="same", activation="relu")

        self.pool5 = MaxPooling2D(strides=(2, 2))
        self.flatten = Flatten()
        self.dense1 = Dense(256, activation="relu", name="dense1")
        self.dense2 = Dense(128, activation="relu", name="dense2")
        self.dense3 = Dense(1, activation="sigmoid", name="dense3")

        self.build((32, 224, 224, 3))
        self.set_weights_to_vgg16()

        # print(vgg16.get_weight_paths().keys())
        # print(vgg16.get_layer("block1_conv1").get_weights())
        # print(vgg16.get_weight_paths())

    def set_weights_to_vgg16(self):
        vgg16 = tf.keras.applications.VGG16(include_top=False)
        self.conv1_1.set_weights(vgg16.get_layer("block1_conv1").get_weights())
        self.conv1_1.trainable = False

        self.conv1_2.set_weights(vgg16.get_layer("block1_conv2").get_weights())
        self.conv1_2.trainable = False

        self.conv2_1.set_weights(vgg16.get_layer("block2_conv1").get_weights())
        self.conv2_1.trainable = False

        self.conv2_2.set_weights(vgg16.get_layer("block2_conv2").get_weights())
        self.conv2_2.trainable = False

        self.conv3_1.set_weights(vgg16.get_layer("block3_conv1").get_weights())
        self.conv3_1.trainable = False

        self.conv3_2.set_weights(vgg16.get_layer("block3_conv2").get_weights())
        self.conv3_2.trainable = False

        self.conv3_3.set_weights(vgg16.get_layer("block3_conv3").get_weights())
        self.conv3_3.trainable = False

        self.conv4_1.set_weights(vgg16.get_layer("block4_conv1").get_weights())
        self.conv4_1.trainable = False

        self.conv4_2.set_weights(vgg16.get_layer("block4_conv2").get_weights())
        self.conv4_2.trainable = False

        self.conv4_3.set_weights(vgg16.get_layer("block4_conv3").get_weights())
        self.conv4_3.trainable = False

        self.conv5_1.set_weights(vgg16.get_layer("block5_conv1").get_weights())
        self.conv5_1.trainable = False

        self.conv5_2.set_weights(vgg16.get_layer("block5_conv2").get_weights())
        self.conv5_2.trainable = False

        self.conv5_3.set_weights(vgg16.get_layer("block5_conv3").get_weights())
        self.conv5_3.trainable = False

    @tf.function
    def call(self, x, training=False):
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.pool1(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.pool2(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.pool3(x)
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.pool4(x)
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.pool5(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.dense3(x)