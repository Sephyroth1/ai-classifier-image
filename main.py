import random

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

seed = 42
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train[:9000]
x_test = x_test[:9000]
y_train = y_train[:9000]
y_test = y_test[:9000]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
batch_size = 32
learning_rate = 0.005
dropout_rate = 0.25

datagen = ImageDataGenerator(
    {
        "rotation_range": 15,
        "width_shift_range": 0.1,
        "height_shift_range": 0.1,
        "horizontal_flip": True,
        "zoom_range": 0.1,
        "shear_range": 0.1,
        "fill_mode": "nearest",
    }
)
datagen.fit(x_train)

model = Sequential(
    [
        Input(shape=(32, 32, 3)),
        Conv2D(32, (3, 3), padding="same"),
        BatchNormalization(),
        Activation("relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(dropout_rate),
        Conv2D(64, (3, 3), padding="same"),
        BatchNormalization(),
        Activation("relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(dropout_rate),
        Conv2D(128, (3, 3), padding="same"),
        BatchNormalization(),
        Activation("relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(dropout_rate),
        Flatten(),
        Dense(64),
        BatchNormalization(),
        Activation("relu"),
        Dropout(min(dropout_rate * 2, 0.6)),
        Dense(10, activation="softmax"),
    ]
)

model.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.fit(
    datagen.flow(x_train, y_train, batch_size=batch_size, seed=seed),
    epochs=40,
    validation_data=(x_test, y_test),
    callbacks=[
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    ],
)

model.evaluate(x_test, y_test)

model.save("model.keras")
