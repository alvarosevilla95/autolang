FROM python:3.11

ARG openai_key

ENV PYTHONBUFFERED 1
ENV OPENAI_API_KEY $openai_key

RUN python -m pip install --upgrade pip

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt 

ADD autolang autolang
CMD [ "python3", "-u", "-m" , "autolang"]
