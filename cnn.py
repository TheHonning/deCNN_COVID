import tensorflow as tf
from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import Adam
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten
import matplotlib.pyplot as plt
import preprocess

class SarsCovCNN(tf.keras.Model):
    
    def __init__(self, img_height:int = 300, img_width:int = 300) -> None:
        super(SarsCovCNN, self).__init__()

        input_shape = (img_height, img_width, 3)
        kernel_sizes = [(3 * 3), (6 * 6), (9 * 9)]
        #pooling_factors = [2, 4 ,6, 8, 10]

        self.C1 = Conv2D(16, kernel_sizes[0], padding='same', input_shape = input_shape, activation="relu")
        self.P1 = MaxPool2D()
        self.C2 = Conv2D(32, kernel_sizes[1], padding='same', activation="relu")
        self.P2 = MaxPool2D()
        self.C3 = Conv2D(32, kernel_sizes[2], padding="same", activation="relu")
        self.P3 = MaxPool2D()
        self.F1 = Flatten()
        self.D1 = Dense(128, activation="relu")
        self.D2 = Dense(2, activation="softmax")
    
    @tf.function
    def call(self, x):
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
    
def plot_metrics(history):       
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(history.history['accuracy']))

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.grid(True)
    plt.show()

# Initiate epochs and learning rate as global variables
epochs = 2
lr = 1e-2
batch_size= 32

mymodel = SarsCovCNN()

loss = SparseCategoricalCrossentropy(from_logits=True)
opti = Adam(learning_rate=lr)

train_ds, val_ds = preprocess.preprocess_data(batch_size=batch_size)

mymodel.compile(loss=loss,
                optimizer=opti,
                metrics=["accuracy"])  # for accuracy - instead of tf.keras.metrics.MeanAbsoluteError()

# save logs with Tensorboard
#EXPERIMENT_NAME = "CNN_LSTM"
#current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#logging_callback = tf.keras.callbacks.TensorBoard(log_dir=f".Homework/07/logs/{EXPERIMENT_NAME}/{current_time}")

history = mymodel.fit(train_ds,
                      validation_data=val_ds,
                      epochs=epochs,
                      batch_size=batch_size)
plot_metrics(history)
