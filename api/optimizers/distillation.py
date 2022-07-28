import os
import time

from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Activation, Dropout, BatchNormalization, Dense, Conv2D, MaxPooling2D, Flatten
from optimizers.optimizer import Optimizer


# from utils.distiller_utils import Distiller


class Distillation(Optimizer):

    def __init__(self, project_path, baseline_accuracy, epoch, batch_size, learning_rate, optimizer, color_scheme):
        super().__init__(
            project_path, baseline_accuracy,
            epoch, batch_size, learning_rate, optimizer, color_scheme, 'distillation')
        self.create_model()
        print('Distillation initialized')

    def create_model(self):
        print("Created teacher model")
        print(self.input_shape)
        print(self.X_train.shape)
        self.model = Sequential([
            Conv2D(16, kernel_size=(3, 3), padding='same',
                   input_shape=self.input_shape),
            Activation('relu'),
            Conv2D(16, kernel_size=(3, 3), padding='same'),
            Activation('relu'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2), strides=2),
            Dropout(0.2),
            Conv2D(32, kernel_size=(3, 3)),
            Activation('relu'),
            Conv2D(32, kernel_size=(3, 3)),
            Activation('relu'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2), strides=2),
            Dropout(0.3),
            Flatten(),
            Dense(64),
            Activation('relu'),
            BatchNormalization(),
            Dropout(0.7),
            Dense(10),
            Activation('softmax'),
        ])

        self.model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
            optimizer=self.optimizer
        )

        self.create_distiller()

    def create_distiller(self):
        self.distiller = Distiller(self.model, self.baseline_model)
        print("Distiller model created")

        self.distiller.compile(
            optimizer=self.optimizer,
            metrics=[keras.metrics.SparseCategoricalAccuracy()],
            student_loss_fn=keras.losses.SparseCategoricalCrossentropy(
                from_logits=True),
            distillation_loss_fn=keras.losses.KLDivergence(),
            alpha=0.3,
            temperature=3
        )
        print("Distiller model compiled")
        self.compile_run()

    def compile_run(self):
        training_st = time.process_time()
        self.hist = self.distiller.fit(
            self.X_train, self.y_train,
            batch_size=self.batch_size,
            epochs=self.epoch,
            verbose=2
        )
        training_et = time.process_time()
        self.training_time = training_et - training_st

        self.save_model()

    def get_metrics(self):
        test_evaluation = self.model.evaluate(
            self.X_test, self.y_test, verbose=0)

        inference_st = time.process_time()
        _ = self.model.predict(self.X_test[:1])
        inference_et = time.process_time()

        self.metrics = {
            "hist": self.hist.history,
            "train loss": self.hist.history['student_loss'][-1],
            "train accuracy": self.hist.history['sparse_categorical_accuracy'][-1],
            "test loss": test_evaluation[0],
            "test accuracy": test_evaluation[1],
            "baseline accuracy": self.baseline_accuracy,
            "inference time": inference_et - inference_st,
            "training time": self.training_time,
            "baseline parameters": self.baseline_model.count_params(),
            "parameters": self.model.count_params(),
            "baseline model size": self.get_model_size(self.baseline_model_path),
            "distilled model size": self.get_model_size(
                os.path.join(self.project_path, 'distilled_model.h5')
            )
        }

        return self.metrics

    def save_model(self):
        self.model.save(os.path.join(
            self.project_path, 'distilled_model.h5'))


class Distiller(keras.Model):

    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.teacher = teacher
        self.student = student

    def compile(
            self,
            optimizer,
            metrics,
            student_loss_fn,
            distillation_loss_fn,
            alpha=0.1,
            temperature=3,
    ):

        super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.temperature = temperature
        self.alpha = alpha

    def train_step(self, data):
        x, y = data

        # Forward pass of teacher
        teacher_prediction = self.teacher(x, training=False)
        print("Teacher prediction   ...", teacher_prediction)
        with tf.GradientTape() as tape:

            # Forward pass of student
            student_predcition = self.student(x, training=True)
            # Compute losses
            student_loss = self.student_loss_fn(y, student_predcition)

            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_prediction/self.temperature, axis=1),
                tf.nn.softmax(student_predcition/self.temperature, axis=1)
            )
            loss = self.alpha * student_loss + \
                (1-self.alpha) * distillation_loss
            print("Loss in distiller :", loss)
            # Compute gradients
            trainable_vars = self.student.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)
            gradients = [gradient * (self.temperature ** 2)
                         for gradient in gradients]
            # Update weights
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))

            # Update the metrics configured in `compile()`
            self.compiled_metrics.update_state(y, student_predcition)

            # Return a dict of performance
            results = {m.name: m.result() for m in self.metrics}
            results.update({"student_loss": student_loss,
                           "distillation_loss": distillation_loss})
            print("Train...", results)
            return results

    def test_step(self, data):

        # Unpack the data
        x, y = data

        # Compute predictions
        y_prediction = self.student(x, training=False)

        # calculate the loss
        student_loss = self.student_loss_fn(y, y_prediction)

        # Update the metrics.
        self.compiled_metrics.update_state(y, y_prediction)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})
        print("Test...", results)
        return results
