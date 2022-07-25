from tensorflow.keras import Sequential
from tensorflow.keras.layers import Activation, Dropout, BatchNormalization, Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
import tensorflow as tf
import tensorflow_model_optimization as tfmot

from Optimizer import Optimizer


class Pruning(Optimizer):
  
  def __init__(self,epochs,batch_size, learning_rate,input_shape,callbacks):
        super().__init__(epochs,batch_size, learning_rate,input_shape,callbacks)
        self.create_blueprint()

  def create_blueprint(self):
    
    pruned_model = Sequential([
      Conv2D(32, kernel_size=(3, 3),padding='same',input_shape=self.input_shape),
      Activation('relu'),
      Conv2D(32, kernel_size=(3, 3),padding='same'),
      Activation('relu'),
      BatchNormalization(),
      MaxPooling2D(pool_size=(2, 2),strides=2),
      Dropout(0.2),
      Conv2D(64, kernel_size=(3, 3)),
      Activation('relu'),
      Conv2D(64, kernel_size=(3, 3)),
      Activation('relu'),
      BatchNormalization(),
      MaxPooling2D(pool_size=(2, 2),strides=2),
      Dropout(0.3),
      Flatten(),
      Dense(256),
      Activation('relu'),
      BatchNormalization(),
      Dropout(0.7),
      Dense(10),
      Activation('softmax'),

      ])

    pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0,
        final_sparsity=0.75,
        begin_step=0,
        end_step=15000
        )
      }

    self.blueprint = tfmot.sparsity.keras.prune_low_magnitude(pruned_model, **pruning_params) 