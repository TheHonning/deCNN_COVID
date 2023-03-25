from keras.preprocessing import image
import splitfolders
import os
import pathlib


def preprocess_data(path: str = r"",
                    batch_size: int = 32,
                    img_size: tuple = (300, 300),
                    seed=69420)\
            -> tuple([image.DirectoryIterator, image.DirectoryIterator]):
    if not os.path.isabs(path):
        path = os.path.dirname(os.path.abspath(__file__))
        path = os.sep.join([path, "data"])
    
    img_height, img_width = img_size

    data_bs = pathlib.Path(path)
    splitfolders.ratio(data_bs, output='Imgs/',
                       seed=seed, ratio=(0.7, 0.15, 0.15), group_prefix=None)
    data_gen = image.ImageDataGenerator(rescale=1.0 / 255)
    train_ds = \
        data_gen.flow_from_directory('Imgs/train/',
                                     target_size=(img_height, img_width),
                                     class_mode='binary',
                                     batch_size=batch_size,
                                     shuffle=True, subset='training')
    val_ds = \
        data_gen.flow_from_directory('Imgs/val/',
                                     target_size=(img_height, img_width),
                                     class_mode='binary',
                                     batch_size=batch_size,
                                     shuffle=False)

    return train_ds, val_ds


if __name__ == "__main__":
    train_ds, val_ds = preprocess_data(img_size=(224, 224))
    print(val_ds[0][0].shape)
