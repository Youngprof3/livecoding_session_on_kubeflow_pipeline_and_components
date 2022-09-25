
FROM python:3.10.4-slim

WORKDIR /app

RUN pip install pandas sklearn numpy

COPY data.pkl ./app
ADD  data_preprocessing.py ./app
COPY . /app/

CMD [ "python","./data_preprocessing.py"]