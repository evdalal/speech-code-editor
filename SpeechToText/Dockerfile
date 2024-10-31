FROM nvidia/cuda:12.3.0-base-ubuntu22.04 as gpu

WORKDIR /app

RUN apt-get update -y && \
  apt-get install -y python3 python3-pip libcudnn8 libcudnn8-dev libcublas-12-4 portaudio19-dev

RUN pip3 install torch==2.3.0 torchaudio==2.3.0

COPY requirements-gpu.txt /app/requirements-gpu.txt
RUN pip3 install -r /app/requirements-gpu.txt

RUN mkdir browserClientDemo
COPY browserClientDemo/server.py /app/browserClientDemo/server.py
COPY transcriber /app/transcriber
COPY config.py /app/config.py
COPY utils.py /app/utils.py

EXPOSE 9001
ENV PYTHONPATH "${PYTHONPATH}:/app"
RUN export PYTHONPATH="${PYTHONPATH}:/app"
CMD ["python3", "browserClientDemo/server.py"]

# --------------------------------------------

FROM ubuntu:22.04 as cpu

WORKDIR /app

RUN apt-get update -y && \
  apt-get install -y python3 python3-pip portaudio19-dev

RUN pip3 install torch==2.3.0 torchaudio==2.3.0

COPY requirements.txt /app/requirements.txt
RUN pip3 install -r /app/requirements.txt
COPY config.py /app/config.py
COPY utils.py /app/utils.py

EXPOSE 9001
ENV PYTHONPATH "${PYTHONPATH}:/app"
RUN export PYTHONPATH="${PYTHONPATH}:/app"
CMD ["python3", "browserClientDemo/server.py"]
