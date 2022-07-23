from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Activation, Dropout, BatchNormalization, Dense
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
import tensorflow_model_optimization as tfmot

annotate = tfmot.quantization.keras.quantize_annotate_layer

def create_QuantizedModel():
    
    cnn_quantized = Sequential([
    annotate(Conv2D(32, kernel_size=(3, 3),padding='same',input_shape=(28, 28, 1))),
    annotate(Activation('relu')),
    annotate(Conv2D(32, kernel_size=(3, 3),padding='same')),
    annotate(Activation('relu')),
    BatchNormalization(),
    annotate(MaxPooling2D(pool_size=(2, 2),strides=2)),
    Dropout(0.2),
    annotate(Conv2D(64, kernel_size=(3, 3))),
    annotate(Activation('relu')),
    annotate(Conv2D(64, kernel_size=(3, 3))),
    annotate(Activation('relu')),
    BatchNormalization(),
    annotate(MaxPooling2D(pool_size=(2, 2),strides=2)),
    Dropout(0.3),
    Flatten(),
    annotate(Dense(256)),
    annotate(Activation('relu')),
    BatchNormalization(),
    Dropout(0.7),
    annotate(Dense(10)),
    Activation('softmax'),
    ])



cnn_quantized = tfmot.quantization.keras.quantize_apply(cnn_quantized)
cnn_quantized.compile(
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
  optimizer=Adam(lr=10**-3)
)

training_st = time.process_time()