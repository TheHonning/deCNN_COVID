import tensorflow as tf
import tensorflow_datasets as tfds
from keras.preprocessing import image
import splitfolders
import numpy as np
import os
import pathlib

def preprocess_data(path:str = r"", batch_size:int = 32, seed=69420) ->tuple(DirectoryIterator, DirectoryIterator):
    if not os.path.isabs(path):
        path = os.path.dirname(os.path.abspath(__file__))
        path = os.sep.join([path, "data"])
    
    img_height, img_width = 300, 300

    data_bs = pathlib.Path(path)
    splitfolders.ratio(data_bs, output='Imgs/', seed=seed, ratio=(0.7, 0.15, 0.15), group_prefix=None)
    data_gen = image.ImageDataGenerator(rescale=1.0 / 255)
    train_ds = data_gen.flow_from_directory('Imgs/train/', target_size=(img_height, img_width),
                                            class_mode='binary', batch_size=batch_size, shuffle=True,subset='training')
    val_ds = data_gen.flow_from_directory('Imgs/val/', target_size=(img_height, img_width),
                                          class_mode='binary', batch_size=batch_size, shuffle=False)

    return train_ds, val_ds


if __name__ == "__main__":
    train_ds, val_ds = preprocess_data()
    print(val_ds)