
FROM python:3.10.4-slim-buster

WORKDIR /app

RUN pip install pandas sklearn numpy
COPY test_data.pkl ./app
ADD model_testing.py ./app
ADD lr_model.pkl ./app
ADD svm_model.pkl ./app
ADD knn_model.pkl ./app
COPY . /app/

CMD [ "python","./model_testing.py"]
