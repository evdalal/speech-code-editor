# GPU Image
FROM nvidia/cuda:12.4.0-base-ubuntu22.04 as gpu

WORKDIR /app

# Update and install dependencies
RUN apt-get update -y && \
    apt-get install -y python3 python3-pip libcudnn9-cuda-12=9.5.1.17-1 libcublas-12-4 portaudio19-dev

# Install required Python packages
RUN pip3 install torch==2.3.0 torchaudio==2.3.0

# Copy the application files
COPY requirements-gpu.txt /app/requirements-gpu.txt
RUN pip3 install -r /app/requirements-gpu.txt
RUN mkdir /app/browserClientDemo
COPY browserClientDemo/server.py /app/browserClientDemo/server.py
COPY transcriber /app/transcriber
COPY config.py /app/config.py
COPY utils.py /app/utils.py

# Set environment variables
ENV PYTHONPATH="${PYTHONPATH}:/app"
RUN echo 'export PYTHONPATH="${PYTHONPATH}:/app"' >> ~/.bashrc

# Adjust LD_LIBRARY_PATH for libcudnn
RUN find / -name 'libcudnn_ops*.so*' -exec echo "Adding {} to LD_LIBRARY_PATH" \; -exec sh -c 'export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$(dirname {})' \;

# Expose required port
EXPOSE 9001

# Command to run the application
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
