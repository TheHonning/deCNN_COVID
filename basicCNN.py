import pathlib

import tensorflow as tf
import numpy as np

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Flatten, BatchNormalization, Dropout, MaxPool2D, MaxPooling2D, Conv2D, Rescaling
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras import optimizers
import matplotlib.pyplot as plt
import pandas as pd
from backprop import DeconvNet


def preprocess(seed:int=69420) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path, pathlib.Path, pathlib.Path]:
    data_dir_covid = pathlib.Path('data/COVID')
    data_dir_non_covid = pathlib.Path('data/non-COVID')

    #Using Split-folders to split source folder in 3 categories i.e. Train (70%), Validation(20%) and Test(10%) set.
    #Resource: https://pypi.org/project/split-folders/

    import splitfolders
    splitfolders.ratio("data", output="/dataset",
        seed=seed, ratio=(.7, .2, .1), group_prefix=None, move=False)

    # Define the path for train, validation and test set

    data_dir_train = pathlib.Path('/dataset/train')
    data_dir_val = pathlib.Path('/dataset/val')
    data_dir_test = pathlib.Path('/dataset/test')

    return data_dir_covid, data_dir_non_covid, data_dir_train, data_dir_test, data_dir_val

def data_generator(data_source,img_height:int, img_width:int, batch_size:int=32, seed:int=69420):    
    return keras.utils.image_dataset_from_directory(
        data_source,
        validation_split=None,
        subset=None,
        seed=seed,
        color_mode="grayscale",
        image_size=(img_height, img_width),
        batch_size=batch_size,
        crop_to_aspect_ratio=True,
        shuffle=True
    )

def plot_data(history) -> None:
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    #plt.savefig(f"testing: e={epochs},lr={lr}.png")
    plt.show()

class SarsCovCNN(tf.keras.Model):
    
    def __init__(self, img_height:int = 300, img_width:int = 300) -> None:
        super(SarsCovCNN, self).__init__()


        self.R1 = Rescaling(1./255, input_shape=(img_height, img_width, 1))
        self.C1 = Conv2D(16, 3, padding='same', activation="relu")
        self.P1 = MaxPooling2D()
        self.C2 = Conv2D(32, 3, padding='same', activation="relu")
        self.P2 = MaxPooling2D()
        self.C3 = Conv2D(32, 3, padding="same", activation="relu")
        self.P3 = MaxPooling2D()
        self.F1 = Flatten()
        self.D1 = Dense(128, activation="relu")
        self.D2 = Dense(2, activation="softmax", name="predictions")
    
    @tf.function
    def call(self, x) -> tf.Tensor:
        x = self.R1(x)
        x = self.C1(x)
        x = self.P1(x)
        x = self.C2(x)
        x = self.P2(x)
        x = self.C3(x)
        x = self.P3(x)
        x = self.F1(x)
        x = self.D1(x)
        x = self.D2(x)

        return x


data_dir_covid, data_dir_non_covid, data_dir_train, data_dir_test, data_dir_val = preprocess()

batch_size = 32
img_height = 256
img_width = 256
num_epochs = 5
train_ds = data_generator(data_dir_train, img_height, img_width, batch_size)
val_ds = data_generator(data_dir_val, img_height, img_width, batch_size)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

cnn = SarsCovCNN(img_height=img_height, img_width=img_width)
# cnn = Sequential([
#     Rescaling(1./255, input_shape=(img_height, img_width, 1)),
#     Conv2D(16, 3, activation='relu'),
#     MaxPooling2D(),
#     Conv2D(32, 3, activation='relu'),
#     MaxPooling2D(),
#     Dropout(0.2),
#     Conv2D(64, 3, activation='relu'),
#     MaxPooling2D(),
#     Flatten(),
#     Dropout(0.2),
#     Dense(128, activation='relu'),
#     Dense(2)
# ])

cnn.compile(optimizer='adam',
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

history = cnn.fit(train_ds, batch_size=batch_size, epochs=num_epochs, verbose=1, validation_data=val_ds)
#plot_data(history)

# image = tf.keras.utils.load_img(r"C:\Users\henni\Documents\GitHub\deCNN_COVID\Imgs\val\COVID\Covid (3).png", color_mode="grayscale")
# input_arr = tf.keras.utils.img_to_array(image)
# input_arr = np.array([input_arr])  # Convert single image to a batch.

for image_batch, labels_batch in val_ds:
  X_test = image_batch.numpy()
  y_test = labels_batch.numpy()
  break

preds = cnn.predict(X_test)
masking = np.zeros(preds.shape)
masking[0, np.argmax(preds)] = 1.

deconv = DeconvNet(model=cnn, 
                   layer_name="predictions",
                   input_data=X_test,
                   masking=masking)

heatmap  = deconv.compute()
plt.imshow(heatmap[0], cmap='gray'); plt.axis('off'); plt.show()