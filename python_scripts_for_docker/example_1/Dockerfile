FROM python:3.10.4-slim-buster

WORKDIR /app

ENV PORT 80

COPY hello.py ./app

COPY . /app/

CMD ["python", "./hello.py"]