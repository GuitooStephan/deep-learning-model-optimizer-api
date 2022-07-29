import os
import time
import tensorflow as tf

from utils.data_utils import get_dataset
import tensorflow_model_optimization as tfmot

from tensorflow.keras.optimizers import Adam, SGD, RMSprop


class Optimizer(object):
    def __init__(

            self, project_path, baseline_accuracy, epoch, batch_size,
            learning_rate, optimizer, color_scheme, technique):
        self.project_path = project_path
        self.baseline_accuracy = baseline_accuracy
        self.baseline_model_path = os.path.join(project_path, "model.h5")
        self.data_folder_path = os.path.join(project_path, "data")
        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.color_scheme = color_scheme
        self.technique = technique

        self.set_optimizer(optimizer)
        self.load_baseline_model()
        self.preprocess()

    def create_model(self):
        pass

    def preprocess(self):
        X_train, X_test, X_val, y_train, y_test, y_val = get_dataset(
            self.data_folder_path,
            self.color_scheme
        )
        print(X_train.shape)
        print(y_train.shape)
        print(X_val.shape)
        print(y_val.shape)
        print(X_test.shape)
        print(y_test.shape)

        X_train = X_train.astype('float32')
        X_val = X_val.astype('float32')
        X_test = X_test.astype('float32')

        self.X_train = X_train/255
        self.X_val = X_val/255
        self.X_test = X_test/255
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

        self.set_input_shape()

    def compile_run(self):
        print(self.epoch)
        print(self.batch_size)
        print(self.optimizer)
        print(self.learning_rate)

        self.model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
            optimizer=self.optimizer
        )

        # print('Fitting model...')
        # return None  # Added to avoid training the model - unsupported machine

        training_st = time.process_time()
        self.hist = self.model.fit(
            self.X_train, self.y_train,
            batch_size=self.batch_size,
            epochs=self.epoch,
            verbose=2,
            validation_data=(self.X_val, self.y_val)
        )

        training_et = time.process_time()

        self.training_time = training_et - training_st

    def get_metrics(self):
        test_evaluation = self.model.evaluate(
            self.X_test, self.y_test, verbose=0)

        inference_st = time.process_time()
        _ = self.model.predict(self.X_test[:1])
        inference_et = time.process_time()

        self.metrics = {
            "hist": self.hist.history,
            "train loss": self.hist.history['loss'][-1],
            "train accuracy": self.hist.history['sparse_categorical_accuracy'][-1],
            "validation loss": self.hist.history['val_loss'][-1],
            "validation accuracy": self.hist.history['val_sparse_categorical_accuracy'][-1],
            "test loss": test_evaluation[0],
            "test accuracy": test_evaluation[1],
            "baseline accuracy": self.baseline_accuracy,
            "inference time": inference_et - inference_st,
            "training time": self.training_time,
            "baseline parameters": self.baseline_model.count_params(),
            "parameters": self.model.count_params(),
            "baseline model size": self.get_model_size(self.baseline_model_path)
        }

    def load_baseline_model(self):
        self.baseline_model = tf.keras.models.load_model(
            self.baseline_model_path)

    def save_model(self):
        pass

    def get_model_size(self, file_path):
        size = os.path.getsize(file_path)
        return size

    def set_optimizer(self, optimizer):
        if optimizer == "Adam":
            self.optimizer = Adam(learning_rate=self.learning_rate)
        elif optimizer == "SGD":
            self.optimizer = SGD(learning_rate=self.learning_rate)
        elif optimizer == "RMSprop":
            self.optimizer = RMSprop(learning_rate=self.learning_rate)
        else:
            raise ValueError("Invalid optimizer")

    def set_input_shape(self):
        if self.color_scheme == "rgb":
            self.input_shape = (
                self.X_train.shape[1], self.X_train.shape[2], self.X_train.shape[3])
        elif self.color_scheme == "grayscale":
            self.input_shape = (
                self.X_train.shape[1], self.X_train.shape[2], 1)
        else:
            raise ValueError("Invalid color scheme")
