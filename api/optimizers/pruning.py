from tensorflow.keras import Sequential
from tensorflow.keras.layers import Activation, Dropout, BatchNormalization, Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
import tensorflow as tf
import tensorflow_model_optimization as tfmot

from Optimizer import Optimizer


class Pruning(Optimizer):

    def __init__(self, epochs, batch_size, learning_rate):
        super().__init__(epochs, batch_size, learning_rate)
        self.create_model()

    def create_model(self):

        pruned_model = Sequential([
            Conv2D(32, kernel_size=(3, 3), padding='same',
                   input_shape=(28, 28, 1)),
            Activation('relu'),
            Conv2D(32, kernel_size=(3, 3), padding='same'),
            Activation('relu'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2), strides=2),
            Dropout(0.2),
            Conv2D(64, kernel_size=(3, 3)),
            Activation('relu'),
            Conv2D(64, kernel_size=(3, 3)),
            Activation('relu'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2), strides=2),
            Dropout(0.3),
            Flatten(),
            Dense(256),
            Activation('relu'),
            BatchNormalization(),
            Dropout(0.7),
            Dense(10),
            Activation('softmax'),

        ])

        self.model = pruned_model

    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=0.75,
            begin_step=0,
            end_step=15000
        )
    }

    pruned_model = tfmot.sparsity.keras.prune_low_magnitude(
        pruned_model, **pruning_params
    )
    pruning_callback = tfmot.sparsity.keras.UpdatePruningStep()

    training_st = time.process_time()
    hist = pruned_model.fit(
        X_train, y_train,
        batch_size=128,
        epochs=40,
        verbose=2,
        validation_data=(X_val, y_val),
        callbacks=pruning_callback
    )
    training_et = time.process_time()


print('Training time ', training_et - training_st)
