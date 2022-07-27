# Optimizer api

An api to facilitate the optimization app

## Dowloading project

```bash
git clone git@github.com:Shoab02/action-learning-api.git
```

## Installation

### Create a conda environmenent and activate it

```bash
# if you already have an environment, skip this step
conda create -n dl python=3.8
conda activate dl
```

### Install the requirements

```bash
pip install -r requirements.txt
```

### Install rabbitmq

for Windows users - https://www.rabbitmq.com/install-windows.html

for Mac users -

```bash
# Open a new terminal
brew install rabbitmq
brew services start rabbitmq
```

### Start fastAPI

```bash
# Open a new terminal
conda activate dl
# Skip this step if you are already in the action-learning-api/api directory
cd action-learning-api/api

uvicorn main:app --host=0.0.0.0 --port=8080
```

### Start the workers

For each worker, open a new terminal, activate the environment and move the action-learning/api directory

```bash
# Worker for quantization
celery -A main.celery worker --pool gevent --loglevel=info -Q quantization -n worker-1 --without-mingle
# Worker for pruning
celery -A main.celery worker --pool gevent --loglevel=info -Q pruning -n worker-2 --without-mingle
# Worker for distillation
celery -A main.celery worker --pool gevent --loglevel=info -Q knowledge-distillation -n worker-3 --without-mingle
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)
