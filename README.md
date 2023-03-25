# Optim

Optim is a web-based program designed to enhance deep learning models by 
reducing their storage requirements and minimizing the time required for 
computation and inference. It employs various techniques such as Pruning, 
Quantization, and Knowledge Distillation to achieve this objective.

## Project Description

Despite offering exceptional accuracy, deep learning models encounter 
significant challenges such as high computational requirements, power and 
hardware costs, and negative environmental impact. To make these models 
more versatile and applicable to a broader range of applications such as 
mobile devices, embedded systems, autonomous agents, and real-time 
applications, reducing computing expenses and storage needs is critical.

This project proposes three compression methods: pruning, quantization, 
and distillation, which enable the compression of deep learning models 
into multiple compressed versions, while still maintaining their 
performance. Simply upload your baseline model, and obtain various 
compressed versions to choose from, enabling you to deploy your model on 
any device with ease.


## Architecture
The product was constructed utilising the **MVP architecture** due to its 
advantages and conveniences. The API was developed using **FastAPI**, 
while the database was built with **PostgreSQL**. Background tasks were 
executed using **Celery**. **Streamlit** was employed for the frontend.

## The repositories
- [**API**](https://github.com/GuitooStephan/deep-learning-model-optimizer-api/tree/master/api)
- [**WEB**](https://github.com/GuitooStephan/deep-learning-model-optimizer-web)





