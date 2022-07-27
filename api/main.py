import uvicorn
from fastapi import FastAPI

from routers import optimizers
from config.celery_utils import create_celery



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
