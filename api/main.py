import uvicorn
from fastapi import FastAPI, Depends, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
# from dotenv import load_dotenv
from pydantic import BaseModel
import Optimizer as  op


app = FastAPI(
    title='DLMOptimizer',
    version='1.0',
    description='Optimization of deep learning models'
)


@app.get('/')
@app.get('/home')
def read_home():
    return {'message': 'Thank you for using our Image classification app'}



@app.post("/predict")
def predict(file_path:str):

    # return {"status": 200,
    #         "message": "Successful"}
    # return FileResponse(file_path+'faces.jpeg')
    return op.Optimizer('abcd',40,90,)