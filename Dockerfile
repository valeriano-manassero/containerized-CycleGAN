FROM nvidia/cuda:9.0-base

COPY ./app /app
WORKDIR /app

RUN apt update && \
    apt install -y curl python3-minimal && \
    apt clean

RUN curl https://bootstrap.pypa.io/get-pip.py | python3 && \
    pip3 install -r requirements.txt

CMD python3 main.py