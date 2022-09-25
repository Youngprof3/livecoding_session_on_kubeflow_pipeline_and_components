
FROM python:3.10.4-slim-buster

WORKDIR /app

RUN pip install pandas sklearn numpy

COPY test_data.pkl ./app
ADD model_evaluation.py ./app
ADD y_pred_lr.pkl ./app
ADD y_pred_svm.pkl ./app
ADD y_pred_knn.pkl ./app
COPY . /app
CMD [ "python","./model_evaluation.py"]
