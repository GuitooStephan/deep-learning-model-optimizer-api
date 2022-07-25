
#THIS IS A TEST FILE. DO NOT CONSIDER

from tensorflow.keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

import tensorflow_model_optimization as tfmot
import Quantization
import tensorflow as tf


# import Pruning 
# import Distillation



class Optimizer:

    
#     epoch= 50
#     batch_size = 50
#     optimizer = "adam"
#     learning_rate = 0.1


#     def __init__(self,model, epoch, batch_size,optimizer, learning_rate):
#          self.model = model
#          self.epoch = epoch
#          self.batch_size = batch_size
#          self.optimizer = optimizer
#          self.learning_rate = learning_rate
#          self.batch_size = batch_size


#     def select_technique(self):
#         # match model:
#         #     case ["Pruning"]:
#         #         Pruning(epoch,batch_size,optimizer,learning_rate)

#         print(self.epoch)
#         print(self.batch_size)
#         print(self.optimizer)
#         print(self.learning_rate)

    def preprocess(X_train, X_test, y_train, X_val, y_val):
        # if (type(X_train) == np.ndarray)
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
       

            
    def compile_run(model, X_train, X_test, X_val, y_val, epochs,batch_size, learning_rate):
        
        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
            optimizer=Adam(lr=learning_rate)
            )

        training_st = time.process_time()

        hist = model.fit(
            train, test,
            batch_size=batch_size,
            epochs=epochs,
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



(X_train,y_train), (X_test,y_test) = fashion_mnist.load_data()
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

print("X_train : {}". format(X_train.shape))
print("X_test : {}". format( X_test.shape))
print("X_val : {}". format( X_val.shape))

X_train, X_test, y_train, X_val, y_val = Optimizer.preprocess(X_train, X_test, y_train, X_val, y_val)


print("X_train : {}". format(X_train.shape))
print("X_test : {}". format( X_test.shape))
print("X_val : {}". format( X_val.shape))



cnn_quantized = Quantization.create_QuantizedModel()

cnn_quantized = tfmot.quantization.keras.quantize_apply(cnn_quantized)
hist,training_time = Optimizer.compile_run(cnn_quantized,X_train, X_test, y_train, X_val, y_val,40,128,10**-3)
print('Training time ', training_time)
Optimizer.calculate_loss(hist)

cnn_pruned = Pruning.create_PrunedModel()
hist,training_time = Optimizer.compile_run(cnn_quantized,X_train, X_test, y_train, X_val, y_val,40,128,10**-3)
print('Training time ', training_time)
Optimizer.calculate_loss(hist)

