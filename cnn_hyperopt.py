import tensorflow_datasets as tfds
from hyperopt import tpe, hp, fmin, STATUS_OK, Trials
import pickle
import keras
import numpy as np


class LetterRecognitionModelHyperOpt:
    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.hyperopt_progress_filename = "model.hyperopt"

    @staticmethod
    def __get_model(params):
        model = keras.Sequential()

        if params['model_choice'] == 'one':
            model.add(keras.layers.Conv2D(16, kernel_size=3, activation='relu', padding='same',
                                             input_shape=(28, 28, 1), data_format='channels_last'))
            model.add(keras.layers.Conv2D(16, kernel_size=3, activation='relu', padding='same'))
            model.add(keras.layers.MaxPooling2D(pool_size=2, strides=2))
            model.add(keras.layers.Dropout(params['dropout']))
            model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu'))
            model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu'))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.MaxPooling2D(pool_size=2, strides=2))
            model.add(keras.layers.Dropout(params['dropout_1']))

        elif params['model_choice'] == 'two':
            model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu', padding='same',
                                             input_shape=(28, 28, 1), data_format='channels_last'))
            model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu', padding='same'))
            model.add(keras.layers.MaxPooling2D(pool_size=2, strides=2))
            model.add(keras.layers.Dropout(params['dropout_2']))
            model.add(keras.layers.Conv2D(64, kernel_size=3, activation='relu'))
            model.add(keras.layers.Conv2D(64, kernel_size=3, activation='relu'))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.MaxPooling2D(pool_size=2, strides=2))
            model.add(keras.layers.Dropout(params['dropout_3']))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(params['dense'], activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(params['dropout_4']))

        if params['val_choice'] == 'two':
            model.add(keras.layers.Dense(params['dense_1'], activation='relu'))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Dropout(params['dropout_5']))

        model.add(keras.layers.Dense(10, activation='softmax'))

        adam = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                      optimizer=adam)

        return model

    def get_data(self):
        dataset = tfds.load("mnist", as_supervised=True)
        train_data = dataset["train"].as_numpy_iterator()
        test_data = dataset["test"].as_numpy_iterator()
        X_train, y_train = zip(*train_data)
        X_test, y_test = zip(*test_data)
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_test, y_test = np.array(X_test), np.array(y_test)

        self.X_train = X_train.reshape(60000, 28, 28, 1)
        self.X_test = X_test.reshape(10000, 28, 28, 1)

        self.y_train = keras.utils.to_categorical(y_train)
        self.y_test = keras.utils.to_categorical(y_test)

    def train(self, params):
        self.model = self.__get_model(params)
        self.model.fit(self.X_train, self.y_train,
                       batch_size=256,
                       epochs=15,
                       verbose=2,
                       validation_data=(self.X_test, self.y_test))

        score, acc = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        print('Val accuracy:', acc)
        return {'loss': -acc, 'status': STATUS_OK, 'model': self.model}
        # return -acc

    def find_best_model(self, max_evals, trials=None):
        if trials is None:
            trials = Trials()

        space = {
            "model_choice": hp.choice("model_choice", ['one', 'two']),
            "val_choice": hp.choice("val_choice", ['one', 'two']),
            "dropout": hp.uniform("dropout", 0, 1),
            "dropout_1": hp.uniform("dropout_1", 0, 1),
            "dropout_2": hp.uniform("dropout_2", 0, 1),
            "dropout_3": hp.uniform("dropout_3", 0, 1),
            "dropout_4": hp.uniform("dropout_4", 0, 1),
            "dropout_5": hp.uniform("dropout_5", 0, 1),
            "dense": hp.choice("dense", [256, 512, 1024]),
            "dense_1": hp.choice("dense_1", [256, 512, 1024]),
        }

        best = fmin(
            fn=self.train,
            space=space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials
        )

        return best

    def find_best_model_with_persistence(self, max_eval, eval_step):
        try:
            trials = pickle.load(open(self.hyperopt_progress_filename, "rb"))
            current_progress = len(trials.trials)

        except:
            trials = Trials()
            current_progress = 0

        best = None
        while current_progress < max_eval:
            current_progress = current_progress + eval_step
            if current_progress > max_eval:
                current_progress = max_eval

            best = self.find_best_model(current_progress, trials)

            with open(self.hyperopt_progress_filename, "wb") as f:
                pickle.dump(trials, f)

        # Best: {
        #     'val_choice': 0,
        #     'dense': 0,
        #     'dense_1': 0,
        #     'dropout': 0.5784593140065394,
        #     'dropout_1': 0.29013743354943095,
        #     'dropout_2': 0.16102854455778437,
        #     'dropout_3': 0.5736088850494538,
        #     'dropout_4': 0.21234561132598592,
        #     'dropout_5': 0.6923310182092806,
        #     'model_choice': 1,
        # }
        return best
