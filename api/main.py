from fastapi import FastAPI, Depends, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
import uvicorn
import os
import cv2
import glob
import numpy as np


from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import fashion_mnist
from Optimizer import Optimizer
from techniques.Quantization import Quantization
import tensorflow_model_optimization as tfmot


app = FastAPI(
    title='DLMOptimizer',
    version='1.0',
    description='Optimization of deep learning models'
)


@app.get('/')
@app.get('/home')
def read_home():
    return {'message': 'Thank you for using our Image classification app'}

@app.post("/read")
def read(file_path:str):

    X_train=[]
    files = glob.glob(file_path+"/train/*")
    for myFile in files:
        image = cv2.imread(myFile)
        image = cv2.resize(image,(32,32))
        X_train.append(image)

    X_test=[]
    files = glob.glob(file_path+"/test/*")
    for myFile in files:
        image = cv2.imread(myFile)
        image = cv2.resize(image,(32,32))
        X_test.append(image)
    
    X_val=[]
    files = glob.glob(file_path+"/val/*")
    for myFile in files:
        image = cv2.imread(myFile)
        image = cv2.resize(image,(32,32))
        X_val.append(image)

    X_train=np.asarray(X_train)
    X_test=np.asarray(X_test)
    X_val=np.asarray(X_val)

    print(X_train.shape)
    print(X_test.shape)
    print(X_val.shape)


    # return {"X_train":np.asarray(X_train).shape,
    #         "X_test":np.asarray(X_test).shape,
    #         "X_val":np.asarray(X_val).shape
                # }
    return {"Success":"Success"}






@app.post("/Optimize")
def optimize(file_path:str):

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

    



if __name__ == '__main__':
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)