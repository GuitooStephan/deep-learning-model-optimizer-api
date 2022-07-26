import os
import time
import tensorflow as tf

from utils.data_utils import get_dataset

from tensorflow.keras.optimizers import Adam, SGD, RMSprop


class Optimizer:
    def __init__(self, project_path, baseline_accuracy, epoch, batch_size, learning_rate, optimizer):
        self.project_path = project_path
        self.baseline_accuracy = baseline_accuracy
        self.baseline_model_path = os.path.join(project_path, "model.h5")
        self.data_folder_path = os.path.join(project_path, "data")
        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.set_optimizer(optimizer)
        self.load_baseline_model()
        self.preprocess()

    def create_model(self):
        pass

    def preprocess(self):
        X_train, X_test, X_val, y_train, y_test, y_val = get_dataset(
            self.project_path
        )
        print(X_train.shape)
        print(X_val.shape)
        print(X_test.shape)

        X_train = X_train.astype('float32')
        X_val = X_val.astype('float32')
        X_test = X_test.astype('float32')

        self.X_train = X_train/255
        self.X_val = X_val/255
        self.X_test = X_test/255
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test

        X_train_shape = self.X_train.shape
        self.input_shape = (
            X_train_shape[1], X_train_shape[2], X_train_shape[3]
        )

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

        print("compiled")

        print('Fitting model...')
        return None  # Added to avoid training the model - unsupported machine

        training_st = time.process_time()
        self.hist = self.model.fit(
            X_train, X_test,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=2,
            validation_data=(X_val, y_val)
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
