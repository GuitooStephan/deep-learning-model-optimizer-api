from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import fashion_mnist
import tensorflow_model_optimization as tfmot

from Optimizer import Optimizer
from Quantization import Quantization


#Loading data and splitting to training and test datasets
(X_train,y_train), (X_test,y_test) = fashion_mnist.load_data()
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)


#Creating a model of the chosen optimizer

cnn_quantized = Quantization(epochs = 50,batch_size=120,learning_rate=10**-3)
cnn_quantized.preprocess(X_train, X_test, y_train, X_val, y_val)
# cnn_quantized.create_model()


#Compiling and running the model
hist,training_time = cnn_quantized.compile_run(X_train, X_test, X_val, y_val)
print('Training time ', training_time)

#Printing the losses
cnn_quantized.calculate_loss(hist)
