import Pruning 
import Quantization
import Distillation

class Optimizer:

    
    epoch= 50
    batch_size = 50
    optimizer = "adam"
    learning_rate = 0.1


    def __init__(self,model, epoch, batch_size,optimizer, learning_rate):
         self.model = model
         self.epoch = epoch
         self.batch_size = batch_size
         self.optimizer = optimizer
         self.learning_rate = learning_rate
         self.batch_size = batch_size


    def select_technique(self):
        # match model:
        #     case ["Pruning"]:
        #         Pruning(epoch,batch_size,optimizer,learning_rate)

        print(self.epoch)
        print(self.batch_size)
        print(self.optimizer)
        print(self.learning_rate)

    def preprocess(train, test, val):
        # if (type(X_train) == np.ndarray)
        X_train = train.reshape(48000, 784)
        X_val = val.reshape(12000, 784)
        X_test = test.reshape(10000, 784)
        X_train = train.astype('float32')
        X_val = val.astype('float32')
        X_test = test.astype('float32')
        X_train /= 255
        X_val /= 255
        X_test/= 255

        return [X_train, X_val, X_test ]
            
    def compile_run(model, train, test, X_val, y_val, epochs,batch_size, learning_rate):
        
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




