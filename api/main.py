from fastapi import FastAPI, Depends, UploadFile
import uvicorn
import os
import cv2
import glob
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
# from tensorflow.keras.datasets import fashion_mnist
import tensorflow_model_optimization as tfmot

from Optimizer import Optimizer
from techniques.Quantization import Quantization
from techniques.Pruning import Pruning

app = FastAPI(
    title='DLMOptimizer',
    version='1.0',
    description='Optimization of deep learning models'
)


@app.get('/')
@app.get('/home')
def read_home():
    return {'message': 'Thank you for using our classification Neural network optimization app'}

@app.post("/read")
def read(file_path:str):
    pass

                                                                                                                                                                                                                                                                                                                                


@app.post("/Optimize")
def optimize(file_path:str):

    #Extracting files (Datasets) from the filepath

    X_train=[]
    files = glob.glob(os.path.join(file_path,"train/*"))
    for myFile in files:
        image = cv2.imread(myFile)
        image = cv2.resize(image,(32,32))
        X_train.append(image)

    X_test=[]
    files = glob.glob(os.path.join(file_path,"test/*"))
    for myFile in files:
        image = cv2.imread(myFile)
        image = cv2.resize(image,(32,32))
        X_test.append(image)
    
    X_val=[]
    files = glob.glob(os.path.join(file_path,"val/*"))
    for myFile in files:
        image = cv2.imread(myFile)
        image = cv2.resize(image,(32,32))
        X_val.append(image)

    X_train=np.asarray(X_train)
    X_test=np.asarray(X_test)
    X_val=np.asarray(X_val)

    print(X_test.shape[0])
    print(X_test.shape[1])

    y_test = pd.read_csv(os.path.join(file_path,"labels_test.csv"))
    y_train = pd.read_csv(os.path.join(file_path,"labels_train.csv"))
    y_val = pd.read_csv(os.path.join(file_path,"labels_val.csv"))

    le = preprocessing.LabelEncoder()
    le.fit(y_train)
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)
    y_val = le.transform(y_val)

    # #Loading data and splitting to training and test datasets (SAMPLE MNIST)
    # (X_train,y_train), (X_test,y_test) = fashion_mnist.load_data()
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

    #Creating a model of the chosen optimizer

    cnn_quantized = Quantization(epochs = 50,batch_size=120,learning_rate=10**-3,input_shape=(X_train.shape[1],X_train.shape[2],X_train.shape[3]))
    cnn_quantized.preprocess(X_train, y_train, X_test, y_test , X_val, y_val)

    #Compiling and running the model
    # hist,training_time = cnn_quantized.compile_run()
    # print('Training time ', training_time)

    # #Printing the losses
    # cnn_quantized.calculate_loss(hist)


    # PRUNING MODEL
    pruning_callback = tfmot.sparsity.keras.UpdatePruningStep()
    cnn_pruned= Pruning(epochs = 40,batch_size=128,learning_rate=10**-3,input_shape=(X_train.shape[1],X_train.shape[2],X_train.shape[3]),callbacks=pruning_callback)
    cnn_pruned.preprocess(X_train, y_train, X_test, y_test , X_val, y_val)

    #Compiling and running the model
    hist,training_time = cnn_pruned.compile_run()
    print('Training time ', training_time)
    
    #Printing the losses
    cnn_pruned.calculate_loss(hist)



if __name__ == '__main__':
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)