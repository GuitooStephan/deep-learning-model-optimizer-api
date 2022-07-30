import uvicorn
from fastapi import FastAPI
from dotenv import load_dotenv

from routers import optimizers
from config.celery_utils import create_celery
from db import db_models
from db.database import engine

load_dotenv('./.env')

db_models.Base.metadata.create_all(bind=engine)


def create_app() -> FastAPI:
    current_app = FastAPI(
        title='DLMOptimizer',
        version='1.0',
        description='Optimization of deep learning models'
    )

    # Registering the routers and endpoints
    current_app.include_router(optimizers.router)

    current_app.celery_app = create_celery()
    return current_app


app = create_app()
celery = app.celery_app


@app.get('/')
@app.get('/home')
def read_home():
    return {'message': 'Thank you for using our classification Neural network optimization app'}


if __name__ == '__main__':
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
