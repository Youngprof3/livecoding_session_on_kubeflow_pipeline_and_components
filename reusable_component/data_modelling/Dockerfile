
FROM python:3.10.4-slim-buster

WORKDIR /app

RUN pip install pandas sklearn numpy

COPY train_data.pkl ./app
ADD data_modelling.py ./app
COPY . /app/


CMD [ "python","./data_modelling.py"]