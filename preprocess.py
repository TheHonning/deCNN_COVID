from keras.preprocessing import image
import splitfolders
import os
import pathlib

def preprocess_data(path:str = r"", batch_size:int = 32, split:float=0.7, seed:int = 69420) ->tuple[image.DirectoryIterator, image.DirectoryIterator]:
    """Prepares Sars Cov CT Scan Data to be processed by TF/Keras Model.

    Args:
        path (str, optional): absolute path to the partent directory wich contains the Covid and non-Covid foldres. Default behavior searches the current folder for the data directory.
        batch_size (int, optional): Batch Size wich the dataset is prepeared with. Defauls to 32.
        split (float, optional): Factor to split Data into training and validation. Defaults to 0.7 (70% Train)
        seed (int, optinal): Seed for shuffeling. Defaults to 69420.

    Returns:
        train_ds, val_ds (DirectoryIterator, DirectoryIterator): Train Dataset and Validation dataset.
    """
    if not os.path.isabs(path):
        path = os.path.dirname(os.path.abspath(__file__))
        path = os.sep.join([path, "data"])
    
    img_height, img_width = 300, 300

    data_bs = pathlib.Path(path)
    splitfolders.ratio(data_bs, output='Imgs/', seed=seed, ratio=(split, 0.05, 1-split-0.05), group_prefix=None)
    data_gen = image.ImageDataGenerator(rescale=1.0 / 255)
    train_ds = data_gen.flow_from_directory('Imgs/train/', target_size=(img_height, img_width),
                                            class_mode='binary', batch_size=batch_size, shuffle=True,subset='training')
    val_ds = data_gen.flow_from_directory('Imgs/val/', target_size=(img_height, img_width),
                                          class_mode='binary', batch_size=batch_size, shuffle=False)

    return train_ds, val_ds


if __name__ == "__main__":
    train_ds, val_ds = preprocess_data()
    print(val_ds)