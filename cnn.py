import tensorflow as tf
from keras.datasets import mnist
import numpy as np
import os
import cv2


class LetterRecognitionModel:
    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.model_filename = "model.keras"

    @staticmethod
    def __get_model():
        model = tf.keras.Sequential()

        model.add(tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu', padding='same',
                                         input_shape=(28, 28, 1), data_format='channels_last'))
        model.add(tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu', padding='same'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2))
        model.add(tf.keras.layers.Dropout(0.16102854455778437))
        model.add(tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu'))
        model.add(tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2))
        model.add(tf.keras.layers.Dropout(0.5736088850494538))

        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(256, activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(0.21234561132598592))

        model.add(tf.keras.layers.Dense(10, activation='softmax'))

        adam = tf.keras.optimizers.Adam(lr=0.001)
        model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                      optimizer=adam)

        return model

    def __save_model(self):
        self.model.save(self.model_filename)

    def get_data(self):
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        self.X_train = X_train.reshape(60000, 28, 28, 1)
        self.X_test = X_test.reshape(10000, 28, 28, 1)

        self.y_train = tf.keras.utils.to_categorical(y_train)
        self.y_test = tf.keras.utils.to_categorical(y_test)

    def train(self):
        is_model_exist = os.path.exists(self.model_filename)
        if is_model_exist:
            self.model = tf.keras.models.load_model(self.model_filename)
            return

        self.model = self.__get_model()
        self.model.fit(self.X_train, self.y_train,
                       batch_size=256,
                       epochs=3,
                       verbose=2,
                       validation_data=(self.X_test, self.y_test))
        self.__save_model()

    def predict(self, frame):
        if len(frame[0][0]) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frame = cv2.resize(frame, (28, 28), interpolation=cv2.INTER_LINEAR)
        frame = frame.reshape(28, 28, 1)

        result = self.model.predict(np.array([frame]))
        predicted_class = result.argmax(axis=1)
        return predicted_class
