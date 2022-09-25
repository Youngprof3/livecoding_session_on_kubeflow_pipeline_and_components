
FROM python:3.10.4-slim

WORKDIR /app

RUN pip install pandas
COPY data_loading.py ./app


COPY . /app/


CMD [ "python","./data_loading.py" ]