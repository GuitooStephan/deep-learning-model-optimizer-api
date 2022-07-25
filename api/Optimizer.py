from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
import tensorflow_model_optimization as tfmot
import tensorflow as tf



class Optimizer:

    def __init__(self, epoch,batch_size,learning_rate):

        self.epoch= epoch
        self.batch_size = batch_size
        self.optimizer = "adam"
        self.learning_rate = learning_rate
        

    def create_model(X_train, X_test, y_train, X_val, y_val):
        pass


    def preprocess(self,X_train, X_test, y_train, X_val, y_val):

        X_train = X_train.reshape(48000, 784)
        X_val = X_val.reshape(12000, 784)
        X_test = X_test.reshape(10000, 784)
        X_train = X_train.astype('float32')
        X_val = X_val.astype('float32')
        X_test = X_test.astype('float32')
        X_train /= 255
        X_val /= 255
        X_test/= 255

        nb_classes = 10
        y_train = to_categorical(y_train, nb_classes)
        y_val = to_categorical(y_val, nb_classes)

        return [X_train, X_test, y_train, X_val, y_val]


    def compile_run(self, X_train, X_test, X_val, y_val):

        print(self.epoch)
        print(self.batch_size)
        print(self.optimizer)
        print(self.learning_rate)
        
        self.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
            optimizer=Adam(lr=self.learning_rate)
            )

        print("compiled")
        return None

        training_st = time.process_time()

        hist = self.fit(
            X_train, X_test,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=2,
            validation_data=(X_val, y_val)
        )

        training_et = time.process_time()

        return [hist, (training_et - training_st)]

    def calculate_loss(hist):
        return {
        "Train loss": hist.history['loss'][-1],
        "Train accuracy": hist.history['sparse_categorical_accuracy'][-1],
        "Validation loss": hist.history['val_loss'][-1],
        "Validation accuracy": hist.history['val_sparse_categorical_accuracy'][-1]
        }
