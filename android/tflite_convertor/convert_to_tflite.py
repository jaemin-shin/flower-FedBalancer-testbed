import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from tfltransfer import bases
from tfltransfer import heads
from tfltransfer import optimizers
from tfltransfer.tflite_transfer_converter import TFLiteTransferConverter

"""Define the base model.

To be compatible with TFLite Model Personalization, we need to define a
base model and a head model. 

Here we are using an identity layer for base model, which just passes the 
input as it is to the head model.
"""
base = tf.keras.Sequential(
    [tf.keras.Input(shape=(128, 9)), tf.keras.layers.Lambda(lambda x: x)]
)

base.compile(loss="categorical_crossentropy", optimizer="sgd")
base.save("identity_model", save_format="tf")

"""Define the head model.

This is the model architecture that we will train using Flower. 
"""
head = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(1152)),
        tf.keras.layers.Reshape((128, 9)),
        tf.keras.layers.Conv1D(192, 16, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling1D(pool_size=4),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=256, activation="relu"),
        tf.keras.layers.Dense(units=6, activation="softmax"),
    ]
)

head.compile(loss="categorical_crossentropy", optimizer="sgd")


"""Convert the model for TFLite.

Using 10 classes in CIFAR10, learning rate = 1e-3 and batch size = 32

This will generate a directory called tflite_model with five tflite models.
Copy them in your Android code under the assets/model directory.
"""

base_path = bases.saved_model_base.SavedModelBase("identity_model")
converter = TFLiteTransferConverter(
    6, base_path, heads.KerasModelHead(head), optimizers.SGD(5e-3), train_batch_size=10
)

converter.convert_and_save("tflite_model")
