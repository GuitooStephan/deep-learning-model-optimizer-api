from tensorflow.keras import Sequential, models
from tensorflow.keras.layers import Activation, Dropout, BatchNormalization, Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from tensorflow import keras
import tensorflow as tf
from optimizers.optimizer import Optimizer


class Distillation(Optimizer):
    def __init__(self, project_path, baseline_accuracy, epoch, batch_size, learning_rate, optimizer):
        super().__init__(project_path, baseline_accuracy,
                         epoch, batch_size, learning_rate, optimizer)
        self.create_model()
        print('Distillation initialized')

    def create_model(self):
        self.teacher = tf.keras.models.clone_model(
            self.baseline_model
        )

    def create_student_model(self):

        self.student = Sequential([
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

        return self.student


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
        print("Tecaher prediction   ...", teacher_prediction)
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

    # Initialize  distiller

# distiller= Distiller(student=student, teacher=teacher)
# distiller.compile(
#     optimizer=keras.optimizers.Adam(lr=10**-3),
#     metrics=[keras.metrics.SparseCategoricalAccuracy()],
#     student_loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     distillation_loss_fn=keras.losses.KLDivergence(),
#     alpha=0.3,
#     temperature=3
#     )

# training_st = time.process_time()
# hist = distiller.fit(X_train, y_train,batch_size=128,
# epochs=40,
# verbose=2
# )
# training_et = time.process_time()

# print('Training time ', training_et - training_st)
