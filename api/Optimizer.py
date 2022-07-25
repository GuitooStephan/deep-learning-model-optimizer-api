from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
import tensorflow_model_optimization as tfmot
import tensorflow as tf



class Optimizer:

    def __init__(self, epoch,batch_size,learning_rate,input_shape):

        self.epoch= epoch
        self.batch_size = batch_size
        self.optimizer = "adam"
        self.learning_rate = learning_rate
        self.input_shape = input_shape

    def create_model(X_train, X_test, y_train, y_test, X_val, y_val ):
        pass


    def preprocess(self,X_train,y_train, X_test, y_test , X_val, y_val):
        

        print(X_train.shape)
        print(X_val.shape)
        print(X_test.shape)

        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2],X_train.shape[3])
        X_val = X_val.reshape(X_val.shape[0], X_val.shape[1]*X_val.shape[2],X_val.shape[3])
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1]*X_test.shape[2],X_test.shape[3])
        X_train = X_train.astype('float32')
        X_val = X_val.astype('float32')
        X_test = X_test.astype('float32')
        self.X_train = X_train/255
        self.X_val = X_val/255
        self.X_test= X_test/255   


    def compile_run(self):

        print(self.epoch)
        print(self.batch_size)
        print(self.optimizer)
        print(self.learning_rate)
        
        self.blueprint.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
            optimizer=Adam(lr=self.learning_rate)
            )

        print("compiled")

        return None #Added to avoid training the model - unsupported machine

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
